import os
import openai
from dotenv import load_dotenv
import argparse
import yaml
import logging
import sys

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('openai_image_gen.log')
    ]
)
logger = logging.getLogger(__name__)

# .envからAPIキーを読み込む（環境変数でもOK）
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

def generate_image(prompt, output_path, model="dall-e-3"):
    try:
        response = openai.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        # レスポンスの型や主要情報をログに記録（b64_json本体は出さない）
        logger.info(f"OpenAI API response type: {type(response)}; keys: {list(response.data[0].__dict__.keys()) if hasattr(response.data[0], '__dict__') else dir(response.data[0])}")
        image_url = getattr(response.data[0], 'url', None)
        b64_json = getattr(response.data[0], 'b64_json', None)
        if image_url:
            # 画像をダウンロードして保存
            import requests
            img_data = requests.get(image_url).content
            with open(output_path, 'wb') as f:
                f.write(img_data)
            logger.info(f"Saved: {output_path} (from url)")
        elif b64_json:
            import base64
            img_data = base64.b64decode(b64_json)
            with open(output_path, 'wb') as f:
                f.write(img_data)
            logger.info(f"Saved: {output_path} (from b64_json, length={len(b64_json)})")
        else:
            logger.error(f"No image URL or b64_json returned from OpenAI API. Response keys: {list(response.data[0].__dict__.keys()) if hasattr(response.data[0], '__dict__') else dir(response.data[0])}")
            return
    except Exception as e:
        # openai.BadRequestErrorなどのAPI例外は内容を詳細に記録
        logger.error(f"Error in generate_image: {str(e)}")
        if hasattr(e, 'response'):
            try:
                logger.error(f"OpenAI API error response: {e.response.text}")
            except Exception:
                pass
        raise

def load_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_prompt(global_conf, objects, scene_list):
    # グローバル条件を冒頭に列挙
    global_lines = [str(v) for v in global_conf.values()]
    prompt = "\n".join(global_lines)
    prompt += "\n\nグリッドには以下を配置する。\n"
    # オブジェクト名を詳細説明に置換し、番号付きで列挙
    for idx, scene in enumerate(scene_list, 1):
        replaced = scene
        for name, obj in objects.items():
            desc = '、'.join(obj['description']) if isinstance(obj, dict) and 'description' in obj else '、'.join(obj)
            replaced = replaced.replace(name, desc)
        prompt += f"{idx}. {replaced}\n"
    prompt += "\n"
    return prompt

def generate_scenes_with_gpt(scene_prompts, num):
    # GPT-4oでシーン案を生成
    system_prompt = "あなたはLINEスタンプのシーン案を考えるアシスタントです。各案は日本語で短く、1文で、キャラクターや状況が明確に伝わるようにしてください。"
    user_prompt = "以下の条件を満たすLINEスタンプ用シーン案を{num}個、日本語で短く列挙してください。\n".format(num=num)
    for cond in scene_prompts:
        user_prompt += f"- {cond}\n"
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1024,
        temperature=0.7
    )
    text = response.choices[0].message.content
    lines = [line.strip(" ・-0123456789.") for line in text.splitlines() if line.strip()]
    return lines[:num]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI画像生成スクリプト")
    parser.add_argument('--model', type=str, default="gpt-image-1", help='画像生成モデル名（gpt-image-1, dall-e-3 など）')
    parser.add_argument('--global', '-g', dest='global_config', type=str, default="input/global_config.yaml", help='グローバル設定YAMLファイル')
    parser.add_argument('--objects', '-o', dest='objects_config', type=str, default="input/objects_config.yaml", help='オブジェクト設定YAMLファイル')
    parser.add_argument('--scenes', '-s', type=str, required=True, help='シーン設定YAMLファイル')
    parser.add_argument('--output_dir', type=str, default="output/generated", help='画像出力ディレクトリ')
    parser.add_argument('--prompt_only', '-p', action='store_true', help='画像生成を行わずプロンプトのみ出力')
    args = parser.parse_args()

    # 実行時のオプション情報をログに出力
    logger.info(f"実行オプション: {vars(args)}")

    # 設定ロード
    global_conf = load_yaml(args.global_config)['global']
    objects = load_yaml(args.objects_config)['objects']
    scenes_yaml = load_yaml(args.scenes)
    scenes = scenes_yaml['scenes']

    os.makedirs(args.output_dir, exist_ok=True)

    for scene_key, scene_info in scenes.items():
        mode = scene_info.get('generate_mode', 'user_defined')
        scene_list = scene_info.get('scene', [])
        if mode == 'gpt_generated':
            num = scene_info.get('num', 9)
            # GPT-4oで案を生成
            generated_scenes = generate_scenes_with_gpt(scene_list, num)
            # user_defined YAMLを自動生成
            new_yaml = {
                'scenes': {
                    scene_key: {
                        'generate_mode': 'user_defined',
                        'scene': generated_scenes
                    }
                }
            }
            out_yaml_path = os.path.join(args.output_dir, f"{scene_key}-scene.yaml")
            with open(out_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(new_yaml, f, allow_unicode=True)
            print(f"[GPT生成] {scene_key}-scene.yaml を生成しました")
            # 生成したYAMLを即時ロードしてuser_definedと同じプロンプト生成・画像生成
            loaded = load_yaml(out_yaml_path)
            user_scene_list = loaded['scenes'][scene_key]['scene']
            prompt = build_prompt(global_conf, objects, user_scene_list)
            base_name = f"{scene_key}"
            output_path = os.path.join(args.output_dir, f"{base_name}.png")
            prompt_path = os.path.join(args.output_dir, f"{base_name}.txt")
            print(f"生成プロンプト[{base_name}]:\n{prompt}")
            if not args.prompt_only:
                generate_image(prompt, output_path, model=args.model)
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            continue
        # user_definedの場合は従来通り
        prompt = build_prompt(global_conf, objects, scene_list)
        base_name = f"{scene_key}"
        output_path = os.path.join(args.output_dir, f"{base_name}.png")
        prompt_path = os.path.join(args.output_dir, f"{base_name}.txt")
        print(f"生成プロンプト[{base_name}]:\n{prompt}")
        if not args.prompt_only:
            generate_image(prompt, output_path, model=args.model)
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt) 