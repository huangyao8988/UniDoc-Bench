import os
import sys
import ast
import json
import time
import base64
import argparse
import re
from tqdm import tqdm
import copy
from openai import OpenAI
from utils import flatten_unique_ignore_case
from prompts.query_syn_prompt import obtain_user_prompt
from prompts.templates import CHOOSE_TEMPLATE_PROMPT_USER, CHOOSE_TEMPLATE_PROMPT_SYSTEM, MESSAGE_WITH_EXAMPLE, choose_fixed_templates


def find_fig_tables(text):
    tabs = re.findall(r"<<tab-([^>]+)>>", text)
    figs = re.findall(r"<<fig-([^>]+)>>", text)
    return {"Table": set(tabs), "Figure": set(figs)}


def chunk_match_back(chunk, chunks_metadata, folder_elements):
    """
    chunk: str
    chunks_metadata: dict. {"source: xxx"}
    folder_elements: str: path to elements folder

    output dict: element_id: img_path
    """
    tab_fig_dict = find_fig_tables(chunk)

    elements_dict = dict()

    # 兼容外部检索数据：如果没有 source 字段，从 document_keyword 构造
    file_name = None
    if 'source' in chunks_metadata:
        file_name = os.path.splitext(os.path.basename(chunks_metadata["source"]))[0].split("_id")[0]
    elif 'document_keyword' in chunks_metadata:
        # 从 document_keyword 构造（例如：026_已解密.pdf -> 026_已解密）
        file_name = chunks_metadata["document_keyword"].replace('.pdf', '')

    # 如果没有文件名信息，返回空字典（但不影响后续处理）
    if not file_name:
        if hasattr(args, 'debug') and args.debug:
            print(f"Warning: No file name info found in chunks_metadata")
        return {}, {}

    elements_file_path = os.path.join(folder_elements, file_name + ".json")

    if not os.path.exists(elements_file_path):
        if hasattr(args, 'debug') and args.debug:
            print(f"Warning: Elements file not found: {elements_file_path}")
        return {}, {}

    try:
        with open(elements_file_path, 'r') as f:
            elements = json.load(f)
            for element in elements["elements"]:
                try:
                    element_id = element["element_id"]
                    image_path = element["metadata"]["image_path"]
                    elements_dict[element_id] = image_path
                except Exception as error:
                    pass
    except Exception as error:
        if hasattr(args, 'debug') and args.debug:
            print(f"Warning: Failed to load elements file {elements_file_path}: {error}")
        return {}, {}

    table_paths = dict()
    for table in tab_fig_dict["Table"]:
        if table in elements_dict:
            table_paths[table] = elements_dict[table]

    figure_paths = dict()
    for figure in tab_fig_dict["Figure"]:
        if figure in elements_dict:
            figure_paths[figure] = elements_dict[figure]

    return table_paths, figure_paths


MESSAGE_WITH_EXAMPLE_ONCE = None
MODEL = "gpt-4o"
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)


def load_chunks_from_json(json_path):
    """
    Load chunks from JSON file and classify them by type for each batch result.
    
    Args:
        json_path: Path to the JSON file containing chunks
        
    Returns:
        list: List of dictionaries, each containing chunks_by_type for a batch result
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    batch_chunks_list = []
    
    for batch_result in data.get('batch_results', []):
        chunks = batch_result.get('data', {}).get('chunks', [])
        query_question = batch_result.get('query_question', '')
        
        text_chunks = []
        image_chunks = []
        table_chunks = []
        
        for chunk in chunks:
            doc_type_kwd = chunk.get('doc_type_kwd', '')
            content = chunk.get('content', '')
            document_id = chunk.get('document_id', '')
            
            chunk_info = {
                'content': content,
                'document_id': document_id,
                'query_question': query_question,
                'metadata': chunk
            }
            
            if doc_type_kwd == '':
                text_chunks.append(chunk_info)
            elif doc_type_kwd == 'image':
                if '<table>' in content:
                    table_chunks.append(chunk_info)
                else:
                    image_chunks.append(chunk_info)
        
        chunks_by_type = {
            'text': text_chunks,
            'image': image_chunks,
            'table': table_chunks
        }
        
        batch_chunks_list.append(chunks_by_type)
    
    return batch_chunks_list


def combine_chunks_by_answer_type(batch_chunks_list, distribution, different_file_config):
    """
    Combine chunks by answer type for each batch result.
    
    Args:
        batch_chunks_list: List of dictionaries, each containing chunks_by_type for a batch result
        distribution: Dictionary with answer_type as key and count as value
        different_file_config: Dictionary with answer_type as key and file source rule
        
    Returns:
        dict: Dictionary with answer_type as key and list of chunk groups
    """
    import random
    
    result = {
        'image_as_answer': [],
        'table_as_answer': [],
        'text_as_answer': [],
        'image_plus_text_as_answer': []
    }
    
    for chunks_by_type in batch_chunks_list:
        for answer_type in ['image_as_answer', 'table_as_answer', 'text_as_answer', 'image_plus_text_as_answer']:
            if distribution.get(answer_type, 0) == 0:
                continue
                
            if len(result[answer_type]) >= distribution[answer_type]:
                continue
            
            file_rule = different_file_config.get(answer_type, None)
            
            if answer_type == 'image_as_answer':
                chunks = chunks_by_type['image']
                min_count, max_count = 1, 4
                
            elif answer_type == 'table_as_answer':
                chunks = chunks_by_type['table']
                min_count, max_count = 1, 2
                
            elif answer_type == 'text_as_answer':
                chunks = chunks_by_type['text']
                min_count, max_count = 1, 4
                
            elif answer_type == 'image_plus_text_as_answer':
                text_chunks = chunks_by_type['text']
                image_chunks = chunks_by_type['image']
                min_count, max_count = 2, 5
                
                num_chunks = random.randint(min_count, max_count)
                num_images = random.randint(1, num_chunks - 1)
                num_texts = num_chunks - num_images
                
                selected_images = []
                selected_texts = []
                
                if file_rule == 'different_files_visited':
                    image_docs = set()
                    for img in image_chunks:
                        if img['document_id'] not in image_docs:
                            selected_images.append(img)
                            image_docs.add(img['document_id'])
                            if len(selected_images) >= num_images:
                                break
                    
                    text_docs = set()
                    for txt in text_chunks:
                        if txt['document_id'] not in text_docs:
                            selected_texts.append(txt)
                            text_docs.add(txt['document_id'])
                            if len(selected_texts) >= num_texts:
                                break
                else:
                    selected_images = random.sample(image_chunks, min(num_images, len(image_chunks)))
                    selected_texts = random.sample(text_chunks, min(num_texts, len(text_chunks)))
                
                if len(selected_images) >= 1 and len(selected_texts) >= 1:
                    combined = selected_images[:num_images] + selected_texts[:num_texts]
                    random.shuffle(combined)
                    result[answer_type].append(combined)
                continue
            
            num_chunks = random.randint(min_count, max_count)
            
            if file_rule == 'different_files_visited':
                selected = []
                doc_set = set()
                for chunk in chunks:
                    if chunk['document_id'] not in doc_set:
                        selected.append(chunk)
                        doc_set.add(chunk['document_id'])
                        if len(selected) >= num_chunks:
                            break
            elif file_rule == 'different_files_visited_random':
                selected = random.sample(chunks, min(num_chunks, len(chunks)))
            else:
                if len(chunks) > 0:
                    doc_id = chunks[0]['document_id']
                    same_doc_chunks = [c for c in chunks if c['document_id'] == doc_id]
                    if len(same_doc_chunks) >= num_chunks:
                        selected = random.sample(same_doc_chunks, num_chunks)
                    else:
                        selected = same_doc_chunks
                else:
                    selected = []
            
            if len(selected) >= min_count:
                result[answer_type].append(selected)
    
    return result


def test_mode01(batch_chunks_list, chunk_groups_by_answer_type, verbose=True):
    """
    Test mode 01: Display chunk combination statistics and output test files.
    
    Args:
        batch_chunks_list: List of dictionaries, each containing chunks_by_type for a batch result
        chunk_groups_by_answer_type: Dictionary with answer_type as key and list of chunk groups
        verbose: If True, print statistics; if False, only output files
    """
    if verbose:
        print("=" * 80)
        print("TEST MODE 01: Chunk Combination Statistics")
        print("=" * 80)
        
        text_chunks = []
        image_chunks = []
        table_chunks = []
        
        for chunks_by_type in batch_chunks_list:
            text_chunks.extend(chunks_by_type['text'])
            image_chunks.extend(chunks_by_type['image'])
            table_chunks.extend(chunks_by_type['table'])
        
        print("\nAvailable chunks by type:")
        print(f"  Text chunks: {len(text_chunks)}")
        print(f"  Image chunks: {len(image_chunks)}")
        print(f"  Table chunks: {len(table_chunks)}")
        
        print("\nChunk groups by answer_type:")
        for answer_type, chunk_groups in chunk_groups_by_answer_type.items():
            print(f"\n  {answer_type}:")
            print(f"    Number of groups: {len(chunk_groups)}")
            for i, group in enumerate(chunk_groups, 1):
                print(f"    Group {i}: {len(group)} chunks")
                doc_ids = set([chunk['document_id'] for chunk in group])
                print(f"      Document IDs: {doc_ids}")
        
        print("\n" + "=" * 80)
        print("Outputting test files...")
        print("=" * 80)
    
    output_files = {
        'image_as_answer': 'test_image_as_answer.json',
        'table_as_answer': 'test_table_as_answer.json',
        'text_as_answer': 'test_text_as_answer.json',
        'image_plus_text_as_answer': 'test_image_plus_text_as_answer.json'
    }
    
    for answer_type, filename in output_files.items():
        chunk_groups = chunk_groups_by_answer_type.get(answer_type, [])
        output_data = {
            'answer_type': answer_type,
            'num_groups': len(chunk_groups),
            'groups': []
        }
        
        for group in chunk_groups:
            group_data = {
                'num_chunks': len(group),
                'document_ids': list(set([chunk['document_id'] for chunk in group])),
                'chunks': [chunk['metadata'] for chunk in group]
            }
            output_data['groups'].append(group_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"  Output: {filename} ({len(chunk_groups)} groups)")
    
    if verbose:
        print("\n" + "=" * 80)
        print("Test mode 01 completed!")
        print("=" * 80)


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
    chunks_json_path,
    distribution,
    different_file_config
):
    chunks_all, chunks_metadata_all, chunks_overlapped_items = dict(), dict(), dict()

    batch_chunks_list = load_chunks_from_json(chunks_json_path)
    chunk_groups_by_answer_type = combine_chunks_by_answer_type(batch_chunks_list, distribution, different_file_config)

    for answer_type in ['image_as_answer', 'table_as_answer', 'text_as_answer', 'image_plus_text_as_answer']:
        chunk_groups = chunk_groups_by_answer_type.get(answer_type, [])
        chunks_list = []
        chunks_metadata_list = []
        chunks_overlapped_list = []

        for group in chunk_groups:
            chunks = [chunk['content'] for chunk in group]
            chunks_metadata = [chunk['metadata'] for chunk in group]
            chunks_overlapped = [chunk['query_question'] for chunk in group]

            chunks_list.append(chunks)
            chunks_metadata_list.append(chunks_metadata)
            chunks_overlapped_list.append(chunks_overlapped)

        chunks_all[answer_type] = chunks_list
        chunks_metadata_all[answer_type] = chunks_metadata_list
        chunks_overlapped_items[answer_type] = chunks_overlapped_list

    return chunks_all, chunks_metadata_all, chunks_overlapped_items, batch_chunks_list, chunk_groups_by_answer_type


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def choose_templates(chunks, chunks_metadata, domain_name):
    if MESSAGE_WITH_EXAMPLE_ONCE is None:
        messages = [
            {
                "role": "system",
                "content": CHOOSE_TEMPLATE_PROMPT_SYSTEM,
            }
        ]
    else:
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
        try:
            _, img = chunk_match_back(chunk, chunk_metadata, args.folder_elements)
            for im, p in img.items():
                images[f"<<fig-{im}>>"] = [encode_image(p), p]
        except Exception as e:
            if hasattr(args, 'debug') and args.debug:
                print(f"Warning: Failed to load image for chunk: {e}")

    # 如果没有找到图片，添加说明文本
    if not images:
        user_prompt.append({
            "type": "text",
            "text": "\n\nNote: No images or tables were found for the above chunks. Please analyze the text content and select appropriate templates based on the textual descriptions only.",
        })
    else:
        for fig, img in images.items():
            user_prompt += [
                {
                    "type": "text",
                    "text": f"\n\nImage: {fig}",
                },
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
            if hasattr(args, 'debug') and args.debug:
                print(f"Raw response from API: {response[:500]}...")

            # 更健壮的响应解析
            if "```" not in response:
                # 尝试直接解析
                templates = ast.literal_eval(response.strip())
            else:
                # 移除 ```json 或 ``` 标记
                response = response.replace("```json", "```").split("```")
                if len(response) >= 2:
                    templates = ast.literal_eval(response[1].strip())
                else:
                    raise ValueError(f"Invalid response format: {response[:200]}")

            if hasattr(args, 'debug') and args.debug:
                print(f"Templates chosen: {templates}")
            break
        except Exception as e:
            if hasattr(args, 'debug') and args.debug:
                print(f"Warning: Failed to get templates (attempt {j+1}/3): {e}")
                if 'response' in locals():
                    print(f"  Response was: {response[:200]}")
            time.sleep(j+1)
    return templates


def build_prompt(chunks, chunks_metadata, hint, answer_type, query_type, templates):
    images = {}
    tables = {}

    for chunk, chunk_metadata in zip(chunks, chunks_metadata):
        try:
            tab, img = chunk_match_back(chunk, chunk_metadata, args.folder_elements)
            for t, p in tab.items():
                tables[f"<<tab-{t}>>"] = [encode_image(p), p]
            for im, p in img.items():
                images[f"<<fig-{im}>>"] = [encode_image(p), p]
        except Exception as e:
            if hasattr(args, 'debug') and args.debug:
                print(f"Warning: Failed to load image/table for chunk: {e}")

    combined_chunk_message = "\n\n".join(
        [f"**Chunk {i}:**\n\n{chunk}" for i, chunk in enumerate(chunks, start=1)]
    )

    user_prompt = [{"type": "text", "text": combined_chunk_message}]

    # 如果没有表格或图片，添加说明
    if not tables and not images:
        user_prompt.append({
            "type": "text",
            "text": "\n\nNote: The above chunks contain textual descriptions of visual elements (images/tables) but no actual visual files were provided. Please analyze the textual descriptions to answer questions.",
        })

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
    parser.add_argument("--domain_name", type=str, required=True)
    parser.add_argument(
        "--chunks_json_path", type=str, required=True, help="Path to chunks JSON file"
    )
    parser.add_argument(
        "--test", type=str, help="Test mode: mode01 for testing chunk combinations"
    )
    parser.add_argument(
        "--QA", type=str, help="Generate QA for specific answer_type only (e.g., QA=image_as_answer)"
    )
    parser.add_argument(
        "--different_file_image_as_answer", type=str, 
        choices=["different_files_visited", "different_files_visited_random"],
        help="Control file source for image_as_answer chunks"
    )
    parser.add_argument(
        "--different_file_table_as_answer", type=str,
        choices=["different_files_visited", "different_files_visited_random"],
        help="Control file source for table_as_answer chunks"
    )
    parser.add_argument(
        "--different_file_text_as_answer", type=str,
        choices=["different_files_visited", "different_files_visited_random"],
        help="Control file source for text_as_answer chunks"
    )
    parser.add_argument(
        "--different_file_image_plus_text_as_answer", type=str,
        choices=["different_files_visited", "different_files_visited_random"],
        help="Control file source for image_plus_text_as_answer chunks"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with verbose output"
    )
    args = parser.parse_args()

    if args.debug:
        print("=" * 80)
        print("DEBUG MODE ENABLED")
        print("=" * 80)
        print(f"chunks_json_path: {args.chunks_json_path}")
        print(f"folder_elements: {args.folder_elements}")
        print(f"output_file: {args.output_file}")
        print(f"testset_size: {args.testset_size}")
        print(f"domain_name: {args.domain_name}")
        print(f"test mode: {args.test}")
        print(f"QA filter: {getattr(args, 'QA', None)}")
        print("=" * 80)

    if not os.path.exists(args.chunks_json_path):
        print(f"Error: chunks_json_path does not exist: {args.chunks_json_path}")
        sys.exit(1)
    
    if not os.path.exists(args.folder_elements):
        print(f"Warning: folder_elements does not exist: {args.folder_elements}")
        print("Continuing without image/table loading...")
        if args.debug:
            print("This may result in empty output if chunks contain images/tables")


    output_file = open(
        args.output_file,
        "a",
    )

    answer_type_distribution = choose_distribution(args.testset_size)
    
    different_file_config = {
        'image_as_answer': getattr(args, 'different_file_image_as_answer', None),
        'table_as_answer': getattr(args, 'different_file_table_as_answer', None),
        'text_as_answer': getattr(args, 'different_file_text_as_answer', None),
        'image_plus_text_as_answer': getattr(args, 'different_file_image_plus_text_as_answer', None)
    }
    
    chunks_all, chunks_metadata_all, chunks_overlapped_items, batch_chunks_list, chunk_groups_by_answer_type = load_chunks(
        args.chunks_json_path,
        answer_type_distribution,
        different_file_config
    )

    if args.test in ["mode01", "mode02"]:
        verbose = (args.test == "mode01")
        test_mode01(batch_chunks_list, chunk_groups_by_answer_type, verbose=verbose)
        if args.test == "mode01":
            output_file.close()
            sys.exit(0)

    qa_filter = args.QA if hasattr(args, 'QA') and args.QA else None
    answer_types_to_process = [
        "image_as_answer",
        "image_plus_text_as_answer",
        "text_as_answer",
        "table_as_answer",
    ]
    
    if qa_filter:
        answer_types_to_process = [qa_filter] if qa_filter in answer_types_to_process else answer_types_to_process

    # 统计每种 answer_type 的处理情况
    processing_stats = {
        'image_as_answer': {'total': 0, 'template_fail': 0, 'api_fail': 0, 'parse_fail': 0, 'success': 0, 'failures': []},
        'image_plus_text_as_answer': {'total': 0, 'template_fail': 0, 'api_fail': 0, 'parse_fail': 0, 'success': 0, 'failures': []},
        'text_as_answer': {'total': 0, 'template_fail': 0, 'api_fail': 0, 'parse_fail': 0, 'success': 0, 'failures': []},
        'table_as_answer': {'total': 0, 'template_fail': 0, 'api_fail': 0, 'parse_fail': 0, 'success': 0, 'failures': []},
    }

    lines = 0
    for answer_type in answer_types_to_process:

        chunks_answer, chunks_metadata_answer, hints_answer = (
            chunks_all[answer_type],
            chunks_metadata_all[answer_type],
            chunks_overlapped_items[answer_type],
        )

        for chunks, chunks_metadata, hints in tqdm(
            zip(chunks_answer, chunks_metadata_answer, hints_answer), desc=f"Generating for {answer_type}"
        ):
            processing_stats[answer_type]['total'] += 1
            hints = flatten_unique_ignore_case(hints)
            try:
                templates = choose_templates(chunks, chunks_metadata, args.domain_name)
                templates = [templates[-1]]
            except Exception as e:
                processing_stats[answer_type]['template_fail'] += 1
                processing_stats[answer_type]['failures'].append({
                    'stage': 'choose_templates',
                    'error': str(e),
                    'chunks_preview': chunks[0][:100] if chunks else ''
                })
                if hasattr(args, 'debug') and args.debug:
                    print(f"Warning: Failed to choose templates: {e}")
                templates = []

            if not templates:
                if hasattr(args, 'debug') and args.debug:
                    print(f"Skipping chunk group: no templates available")
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
                    response_raw = (
                        client.chat.completions.create(
                            model=MODEL,
                            messages=messages,
                            temperature=0.85,
                        )
                        .choices[0]
                        .message.content
                    )
                    if hasattr(args, 'debug') and args.debug:
                        print(f"Raw response from API: {response_raw[:500]}...")

                    # 更健壮的响应解析
                    if "```" not in response_raw:
                        response_clean = response_raw.strip()
                    else:
                        response = response_raw.replace("```json", "```").split("```")
                        if len(response) >= 2:
                            response_clean = response[1].strip()
                        else:
                            raise ValueError(f"Invalid response format: {response_raw[:200]}")

                    # 移除可能存在的 <format> 标签
                    if response_clean.startswith("<format>"):
                        response_clean = response_clean.split("</format>", 1)[-1].strip()

                    if hasattr(args, 'debug') and args.debug:
                        print(f"Cleaned response: {response_clean[:200]}...")

                except Exception as error:
                    processing_stats[answer_type]['api_fail'] += 1
                    processing_stats[answer_type]['failures'].append({
                        'stage': 'api_call',
                        'error': str(error),
                        'template_idx': idx
                    })
                    if hasattr(args, 'debug') and args.debug:
                        print(f"Warning: API call failed: {error}")
                    time.sleep(idx+1)
                    continue

                try:
                    questions = ast.literal_eval(response_clean)
                    processing_stats[answer_type]['success'] += 1
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
                        if hasattr(args, 'debug') and args.debug:
                            print(f"Written QA #{lines} to output file")

                except Exception as error:
                    processing_stats[answer_type]['parse_fail'] += 1
                    processing_stats[answer_type]['failures'].append({
                        'stage': 'parse_response',
                        'error': str(error),
                        'response_preview': response_clean[:200] if 'response_clean' in locals() else ''
                    })
                    if hasattr(args, 'debug') and args.debug:
                        print(f"Warning: Failed to process response: {error}")
                    pass

    # 输出统计报告
    print("\n" + "=" * 80)
    print("PROCESSING STATISTICS REPORT")
    print("=" * 80)

    for atype, stats in processing_stats.items():
        print(f"\n{atype}:")
        print(f"  Total groups: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed (choose_templates): {stats['template_fail']}")
        print(f"  Failed (API call): {stats['api_fail']}")
        print(f"  Failed (parse response): {stats['parse_fail']}")

        if stats['failures'] and hasattr(args, 'debug') and args.debug:
            print(f"\n  Failure details (first 5):")
            for i, fail in enumerate(stats['failures'][:5]):
                print(f"    {i+1}. Stage: {fail['stage']}, Error: {fail.get('error', 'N/A')}")
                if 'chunks_preview' in fail:
                    print(f"       Chunks preview: {fail['chunks_preview']}")
                if 'response_preview' in fail:
                    print(f"       Response preview: {fail['response_preview']}")

    print("\n" + "=" * 80)

    output_file.close()

