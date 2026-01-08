import os
import ast
import json
import time
import base64
import argparse
from tqdm import tqdm
import copy
from openai import OpenAI
from utils import flatten_unique_ignore_case
from prompts.query_syn_prompt import obtain_user_prompt
from prompts.templates import CHOOSE_TEMPLATE_PROMPT_USER, MESSAGE_WITH_EXAMPLE, choose_fixed_templates

MESSAGE_WITH_EXAMPLE_ONCE = copy.deepcopy(MESSAGE_WITH_EXAMPLE)
MODEL = "gpt-4o"
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)


def is_text_chunk(chunk):
    """判断是否为文本类型chunks"""
    return chunk.get("doc_type_kwd", "") == ""


def is_image_chunk(chunk):
    """判断是否为图片类型chunks"""
    return chunk.get("doc_type_kwd", "") == "image" and "<table>" not in chunk.get("content", "")


def is_table_chunk(chunk):
    """判断是否为表格类型chunks"""
    return chunk.get("doc_type_kwd", "") == "image" and "<table>" in chunk.get("content", "")


def read_json_file(json_path):
    """读取JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_chunks_by_answer_type(chunks, answer_type, different_files_visited=False):
    """根据answer_type筛选符合条件的chunks"""
    # 检查chunks是否符合类型要求
    chunk_types = []
    for chunk in chunks:
        if is_text_chunk(chunk):
            chunk_types.append("text")
        elif is_table_chunk(chunk):
            chunk_types.append("table")
        elif is_image_chunk(chunk):
            chunk_types.append("image")
        else:
            chunk_types.append("unknown")
    
    # 检查是否来自同一文件
    document_ids = set(chunk["document_id"] for chunk in chunks)
    same_document = len(document_ids) == 1
    
    # 根据answer_type进行筛选
    if answer_type == "image_as_answer":
        # 纯图片答案：仅包含图片类型chunks，无表格、文本；所有chunks来自同一document_id；chunks数量：1-4个
        if all(ct == "image" for ct in chunk_types) and same_document and 1 <= len(chunks) <= 4:
            return True
    elif answer_type == "table_as_answer":
        # 纯表格答案：仅包含表格类型chunks，无图片、文本；可配置是否来自同一document_id；chunks数量：1-2个
        if all(ct == "table" for ct in chunk_types) and 1 <= len(chunks) <= 2:
            if not different_files_visited:
                return same_document
            else:
                return True
    elif answer_type == "text_as_answer":
        # 纯文本答案：仅包含文本类型chunks，无图片、表格；可配置是否来自同一document_id；chunks数量：1-4个
        if all(ct == "text" for ct in chunk_types) and 1 <= len(chunks) <= 4:
            if not different_files_visited:
                return same_document
            else:
                return True
    elif answer_type == "image_plus_text_as_answer":
        # 图文混合答案：必须包含文本和图片，无表格；可配置是否来自同一document_id；chunks数量：1-5个
        if "table" not in chunk_types and "text" in chunk_types and "image" in chunk_types and 1 <= len(chunks) <= 5:
            if not different_files_visited:
                return same_document
            else:
                return True
    
    return False


def extract_chunks_from_json(json_data, answer_type, distribution, different_files_visited=False):
    """从JSON数据中提取符合条件的chunks"""
    extracted_chunks = []
    extracted_metadata = []
    extracted_overlapped = []
    
    # 遍历batch_results中的每个子键
    for batch_item in json_data.get("batch_results", []):
        if len(extracted_chunks) >= distribution:
            break
            
        data = batch_item.get("data", {})
        chunks = data.get("chunks", [])
        
        # 根据answer_type筛选符合条件的chunks
        filtered_chunks = []
        for chunk in chunks:
            is_text = is_text_chunk(chunk)
            is_img = is_image_chunk(chunk)
            is_tab = is_table_chunk(chunk)
            
            if answer_type == "text_as_answer" and is_text:
                filtered_chunks.append(chunk)
            elif answer_type == "image_as_answer" and is_img:
                filtered_chunks.append(chunk)
            elif answer_type == "table_as_answer" and is_tab:
                filtered_chunks.append(chunk)
            elif answer_type == "image_plus_text_as_answer" and (is_text or is_img):
                filtered_chunks.append(chunk)
        
        # 如果没有符合条件的chunks，跳过
        if not filtered_chunks:
            continue
        
        # 确定chunks数量范围
        if answer_type == "image_as_answer":
            min_chunks, max_chunks = 1, 4
        elif answer_type == "table_as_answer":
            min_chunks, max_chunks = 1, 2
        elif answer_type == "text_as_answer":
            min_chunks, max_chunks = 1, 4
        elif answer_type == "image_plus_text_as_answer":
            min_chunks, max_chunks = 1, 5
        
        # 如果需要来自同一document_id，按document_id分组
        if not different_files_visited:
            from collections import defaultdict
            doc_groups = defaultdict(list)
            for chunk in filtered_chunks:
                doc_groups[chunk["document_id"]].append(chunk)
            
            # 从每个document_id组中提取符合条件的chunks
            for doc_id, doc_chunks in doc_groups.items():
                if len(extracted_chunks) >= distribution:
                    break
                
                # 检查chunks数量是否在范围内
                if min_chunks <= len(doc_chunks) <= max_chunks:
                    # 对于image_plus_text_as_answer，需要同时包含text和image
                    if answer_type == "image_plus_text_as_answer":
                        chunk_types = set()
                        for chunk in doc_chunks:
                            if is_text_chunk(chunk):
                                chunk_types.add("text")
                            elif is_image_chunk(chunk):
                                chunk_types.add("image")
                        if "text" not in chunk_types or "image" not in chunk_types:
                            continue
                    
                    # 提取chunks内容
                    chunk_contents = [chunk["content"] for chunk in doc_chunks]
                    # 生成metadata
                    chunk_metadata = [{"source": chunk["document_id"]} for chunk in doc_chunks]
                    # 生成overlapped_items（默认为空列表）
                    overlapped_items = [[]]
                    
                    extracted_chunks.append(chunk_contents)
                    extracted_metadata.append(chunk_metadata)
                    extracted_overlapped.append(overlapped_items)
        else:
            # 不需要来自同一document_id，直接提取
            if answer_type == "image_plus_text_as_answer":
                # 对于image_plus_text_as_answer，需要跨document组合text和image chunks
                text_chunks = []
                image_chunks = []
                for chunk in filtered_chunks:
                    if is_text_chunk(chunk):
                        text_chunks.append(chunk)
                    elif is_image_chunk(chunk):
                        image_chunks.append(chunk)
                
                # 如果没有足够的text或image chunks，无法组合
                if not text_chunks or not image_chunks:
                    continue
                
                # 组合text和image chunks
                # 策略：从text_chunks和image_chunks中各取一个或多个chunks组合
                # 确保组合后的chunks数量在1-5范围内
                
                # 简单策略：每个组合包含1个text chunk和1个image chunk
                # 可以根据需要扩展为更复杂的组合策略
                for i in range(min(len(text_chunks), len(image_chunks), distribution - len(extracted_chunks))):
                    if len(extracted_chunks) >= distribution:
                        break
                    
                    # 组合1个text chunk和1个image chunk
                    combined_chunks = [text_chunks[i], image_chunks[i]]
                    
                    # 检查组合后的chunks数量是否在范围内
                    if min_chunks <= len(combined_chunks) <= max_chunks:
                        # 提取chunks内容
                        chunk_contents = [chunk["content"] for chunk in combined_chunks]
                        # 生成metadata
                        chunk_metadata = [{"source": chunk["document_id"]} for chunk in combined_chunks]
                        # 生成overlapped_items（默认为空列表）
                        overlapped_items = [[]]
                        
                        extracted_chunks.append(chunk_contents)
                        extracted_metadata.append(chunk_metadata)
                        extracted_overlapped.append(overlapped_items)
            else:
                # 对于其他answer_type，按document_id分组
                from collections import defaultdict
                doc_groups = defaultdict(list)
                for chunk in filtered_chunks:
                    doc_groups[chunk["document_id"]].append(chunk)
                
                # 从document组中提取chunks
                for doc_id, doc_chunks in doc_groups.items():
                    if len(extracted_chunks) >= distribution:
                        break
                    
                    # 检查chunks数量是否在范围内
                    if min_chunks <= len(doc_chunks) <= max_chunks:
                        # 提取chunks内容
                        chunk_contents = [chunk["content"] for chunk in doc_chunks]
                        # 生成metadata
                        chunk_metadata = [{"source": chunk["document_id"]} for chunk in doc_chunks]
                        # 生成overlapped_items（默认为空列表）
                        overlapped_items = [[]]
                        
                        extracted_chunks.append(chunk_contents)
                        extracted_metadata.append(chunk_metadata)
                        extracted_overlapped.append(overlapped_items)
    
    return extracted_chunks[:distribution], extracted_metadata[:distribution], extracted_overlapped[:distribution]


def choose_distribution(testset_size):
    answer_types = [
        "image_as_answer",
        "image_plus_text_as_answer",
        "text_as_answer",
        "table_as_answer",
    ]

    base_size = testset_size // 4
    remainder = testset_size % 4

    distribution = {atype: base_size for atype in answer_types}

    for i in range(remainder):
        distribution[answer_types[i]] += 1

    distribution["image_as_answer"] *= 3  # image_only is HARD to obtain

    return distribution


def load_chunks(
    json_data,
    distribution,
    different_files_in_cluster,
    different_files_visited,
    no_tab_in_chunk_text,
    no_tab_in_chunk_img
):
    chunks_all, chunks_metadata_all, chunks_overlapped_items = dict(), dict(), dict()

    # 纯图片答案
    chunks, chunks_metadata, chunks_overlapped_item = extract_chunks_from_json(
        json_data, "image_as_answer", distribution["image_as_answer"], False
    )
    chunks_all["image_as_answer"] = chunks
    chunks_metadata_all["image_as_answer"] = chunks_metadata
    chunks_overlapped_items["image_as_answer"] = chunks_overlapped_item

    # 纯表格答案
    chunks, chunks_metadata, chunks_overlapped_item = extract_chunks_from_json(
        json_data, "table_as_answer", distribution["table_as_answer"], different_files_visited
    )
    chunks_all["table_as_answer"] = chunks
    chunks_metadata_all["table_as_answer"] = chunks_metadata
    chunks_overlapped_items["table_as_answer"] = chunks_overlapped_item

    # 纯文本答案
    chunks, chunks_metadata, chunks_overlapped_item = extract_chunks_from_json(
        json_data, "text_as_answer", distribution["text_as_answer"], different_files_visited
    )
    chunks_all["text_as_answer"] = chunks
    chunks_metadata_all["text_as_answer"] = chunks_metadata
    chunks_overlapped_items["text_as_answer"] = chunks_overlapped_item

    # 图文混合答案
    chunks, chunks_metadata, chunks_overlapped_item = extract_chunks_from_json(
        json_data, "image_plus_text_as_answer", distribution["image_plus_text_as_answer"], different_files_visited
    )
    chunks_all["image_plus_text_as_answer"] = chunks
    chunks_metadata_all["image_plus_text_as_answer"] = chunks_metadata
    chunks_overlapped_items["image_plus_text_as_answer"] = chunks_overlapped_item

    return chunks_all, chunks_metadata_all, chunks_overlapped_items


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def chunk_match_back(chunk, chunks_metadata, folder_elements):
    """简化版的chunk_match_back函数，返回空的表格和图片字典"""
    return {}, {}


def choose_templates(chunks, chunks_metadata, domain_name):
    messages = copy.deepcopy(MESSAGE_WITH_EXAMPLE_ONCE)
    user_prompt = [
        {
            "type": "text",
            "text": CHOOSE_TEMPLATE_PROMPT_USER.replace(
                "{text_contexts}", json.dumps(chunks, indent=4)).replace(
                "{{TEMPLATES}}", choose_fixed_templates(domain_name)
            )
            + "\n\nThese are the tables and images in the above chunks:",
        }
    ]
    images = {}
    for chunk, chunk_metadata in zip(chunks, chunks_metadata):
        _, img = chunk_match_back(chunk, chunk_metadata, args.folder_elements)
        for im, p in img.items():
            images[f"<<fig-{im}>>"] = [encode_image(p), p]

    for fig, img in images.items():
        user_prompt += [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img[0]}",
                    "name": f"This is the image for {fig} in the above context.",
                },
            }
        ]
    messages.append({"role": "user", "content": user_prompt})
    templates = None
    for j in range(3):
        try:
            response = (
                client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.85,
                )
                .choices[0]
                .message.content
            )
            response = response.replace("```json", "```").split("```")[1]
            templates = ast.literal_eval(response)
            break
        except Exception as e:
            time.sleep(j+1)
    return templates


def build_prompt(chunks, chunks_metadata, hint, answer_type, query_type, templates):
    images = {}
    tables = {}

    for chunk, chunk_metadata in zip(chunks, chunks_metadata):
        tab, img = chunk_match_back(chunk, chunk_metadata, args.folder_elements)
        for t, p in tab.items():
            tables[f"<<tab-{t}>>"] = [encode_image(p), p]
        for im, p in img.items():
            images[f"<<fig-{im}>>"] = [encode_image(p), p]

    combined_chunk_message = "\n\n".join(
        [f"**Chunk {i}:**\n\n{chunk}" for i, chunk in enumerate(chunks, start=1)]
    )

    user_prompt = [{"type": "text", "text": combined_chunk_message}]

    for tab, img in tables.items():
        user_prompt += [
            {
                "type": "text",
                "text": f"Below is the image for the TABLE: {tab} in the above context.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img[0]}",
                    "name": f"This is the image for the TABLE: {tab} in the above context.",
                },
            },
        ]

    for fig, img in images.items():
        user_prompt += [
            {
                "type": "text",
                "text": f"Below is the image for the FIGURE: {fig} in the above context.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img[0]}",
                    "name": f"This is the image for the FIGURE: {fig} in the above context.",
                },
            },
        ]
    messages = []
    for template in templates:
        template = json.dumps(template, indent=4)
        user_query = obtain_user_prompt(answer_type, template, query_type).replace(
            "{{hints}}", "/ ".join(hint)
        )

        user_prompt_each = user_prompt + [{"type": "text", "text": user_query}]

        messages.append(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that responds Python dictionary.",
                },
                {"role": "user", "content": user_prompt_each},
            ]
        )

    return messages, tables, images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name_str", type=str, required=True, help="name of the subdomain"
    )
    parser.add_argument(
        "--folder_elements", type=str, required=True, help="Database folder elements"
    )
    parser.add_argument("--output_file", type=str, required=True, help="output_file")
    parser.add_argument("--testset_size", type=int, required=True, help="# of datasets")
    parser.add_argument(
        "--different_files_in_cluster", action="store_true", help="# of datasets"
    )
    parser.add_argument(
        "--different_files_visited", action="store_true", help="# of datasets"
    )
    parser.add_argument("--no_tab_in_chunk_text", action="store_true", help="# of datasets")
    parser.add_argument("--no_tab_in_chunk_img", action="store_true", help="# of datasets")
    parser.add_argument("--domain_name", type=str, required=True)
    parser.add_argument(
        "--json_path", type=str, required=True, help="Path to batch_retrieve_results.json file"
    )
    parser.add_argument(
        "--test_mode", action="store_true", help="Run in test mode, only test JSON reading and chunks filtering"
    )
    args = parser.parse_args()


    output_file = open(
        args.output_file,
        "a",
    )

    # 读取JSON文件
    json_data = read_json_file(args.json_path)

    answer_type_distribution = choose_distribution(args.testset_size)
    chunks_all, chunks_metadata_all, chunks_overlapped_items = load_chunks(
        json_data,
        answer_type_distribution,
        args.different_files_in_cluster,
        args.different_files_visited,
        args.no_tab_in_chunk_text,
        args.no_tab_in_chunk_img
    )
    
    # 测试模式：只输出chunks筛选结果，不执行后续流程
    if args.test_mode:
        print("=== Test Mode Results ===")
        for answer_type in chunks_all:
            print(f"\n{answer_type}:")
            print(f"  Number of chunks: {len(chunks_all[answer_type])}")
            if chunks_all[answer_type]:
                print(f"  Example chunks: {chunks_all[answer_type][0]}")
        output_file.close()
        exit(0)

    lines = 0
    for answer_type in [
        "image_as_answer",
        "image_plus_text_as_answer",
        "text_as_answer",
        "table_as_answer",
    ]:

        chunks_answer, chunks_metadata_answer, hints_answer = (
            chunks_all[answer_type],
            chunks_metadata_all[answer_type],
            chunks_overlapped_items[answer_type],
        )

        for chunks, chunks_metadata, hints in tqdm(
            zip(chunks_answer, chunks_metadata_answer, hints_answer), desc=f"Generating for {answer_type}"
        ):
            hints = flatten_unique_ignore_case(hints)
            try:
                templates = choose_templates(chunks, chunks_metadata, args.domain_name)
                templates = [templates[-1]]
            except Exception as e:
                templates = []

            if not templates:
                continue

            messages_lst, tables, images = build_prompt(
                chunks,
                chunks_metadata,
                hints,
                answer_type,
                query_type=None,
                templates=templates,
            )

            for idx, messages in enumerate(messages_lst):

                try:
                    response = (
                        client.chat.completions.create(
                            model=MODEL,
                            messages=messages,
                            temperature=0.85,
                        )
                        .choices[0]
                        .message.content
                    )
                    response = (
                        response
                        if "```" not in response
                        else response.replace("```json", "```").split("```")[1]
                    )

                except Exception as error:
                    time.sleep(idx+1)

                try:
                    questions = ast.literal_eval(response.strip())
                    for element in questions["questions"]:
                        element["answer_type"] = answer_type
                        element["contexts"] = chunks
                        element["template"] = templates[idx]
                        element["chunks_metadata"] = chunks_metadata
                        element["hints"] = hints
                        element["tables"] = {t: p[1] for t, p in tables.items()}
                        element["images"] = {t: p[1] for t, p in images.items()}

                        output_file.write(json.dumps(element) + "\n")
                        output_file.flush()
                        lines += 1

                except Exception as error:
                    pass

    output_file.close()

