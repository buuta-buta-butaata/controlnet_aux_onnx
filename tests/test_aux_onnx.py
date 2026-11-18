import os
import glob
import pytest
import numpy as np
import onnxruntime as ort
from PIL import Image


IMAGE_DIR = rf"{os.path.dirname(__file__)}/test_images"
ONNX_DIR = rf"{os.path.dirname(__file__)}/../models"
TEED_WEIGHTS = rf"{os.path.dirname(__file__)}/../models/7_model.pth"

# controlnet_aux から各検出器をインポート
# import時にwarningが出るので、無視する
import warnings
warnings.simplefilter("ignore")
from controlnet_aux import (
    PidiNetDetector,
    HEDdetector,
    TEEDdetector,
    LeresDetector,
    LineartDetector,
    LineartAnimeDetector,
    MidasDetector,
    ZoeDetector,
    NormalBaeDetector,
)
warnings.resetwarnings()

# zoeはtransformersと比較
# from zoe_transformers import ZoeDetector

# 今後モデルを追加する場合はここに追記
from controlnet_aux_onnx import (
    PidiNetDetector as PidiOnnx,
    HEDdetector as HEDOnnx,
    TEEDdetector as TEEDOnnx,
    LeresDetector as LeresOnnx,
    LineartDetector as LineartOnnx,
    LineartCoarseDetector as LineartCoarseOnnx,
    LineartAnimeDetector as LineartAnimeOnnx,
    MidasDetector as MidasOnnx,
    ZoeDetector as ZoeOnnx,
    NormalBaeDetector as NormalBaeOnnx,
)

# ==========================================
# 📋 テスト対象モデルの定義（マトリクス）
# ==========================================
default_required_match_rate = 0.95

default_tolerance = (1, None)
binary_image_tolerance = (255, f"Output: Binary image, Max diff tolerance: 255")
ternary_image_tolerance = (128, f"Output: Ternary image, Max diff tolerance: 128")

default_resolution = 512

MODEL_CONFIGS = [
    # pidinet
    (
        "pidi",
        "pidi.onnx",
        lambda: PidiNetDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: PidiOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        default_tolerance,
        default_required_match_rate,
    ),
    (
        "pidi[safe]",
        "pidi.onnx",
        lambda: PidiNetDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: PidiOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, safe=True, output_type="np",
                                                   detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        ternary_image_tolerance, # safe_step 2 (ternary)
        default_required_match_rate,
    ),
    (
        "pidi[scribble]",
        "pidi.onnx",
        lambda: PidiNetDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: PidiOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, scribble=True, output_type="np",
                                                   detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        binary_image_tolerance,
        default_required_match_rate,
    ),
    (
        "pidi[apply_filter]",
        "pidi.onnx",
        lambda: PidiNetDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: PidiOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, apply_filter=True, output_type="np",
                                                   detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        binary_image_tolerance,
        default_required_match_rate,
    ),
    (
        "pidi[all_filter]",
        "pidi.onnx",
        lambda: PidiNetDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: PidiOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, safe=True, scribble=True, apply_filter=True,
                                                   output_type="np",
                                                   detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        binary_image_tolerance,
        default_required_match_rate,
    ),
    # hed
    (
        "hed",
        "hed.onnx",
        lambda: HEDdetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: HEDOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        default_tolerance,
        default_required_match_rate,
    ),
    (
        "hed[safe]",
        "hed.onnx",
        lambda: HEDdetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: HEDOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, safe=True, output_type="np",
                                                   detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        ternary_image_tolerance, # safe_step 2
        default_required_match_rate,
    ),
    (
        "hed[scribble]",
        "hed.onnx",
        lambda: HEDdetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: HEDOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, scribble=True, output_type="np",
                                                   detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        binary_image_tolerance,
        default_required_match_rate,
    ),
    (
        "teed",
        "teed.onnx",
        lambda: TEEDdetector.from_pretrained(r"./", filename=TEED_WEIGHTS),
        lambda model_dir, sess_options: TEEDOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", safe_steps=0,
                                                   detect_resolution=default_resolution),
        default_tolerance,
        default_required_match_rate,
    ),
    (
        "teed[safe]",
        "teed.onnx",
        lambda: TEEDdetector.from_pretrained(r"./", filename=TEED_WEIGHTS),
        lambda model_dir, sess_options: TEEDOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", safe_steps=4,
                                                   detect_resolution=default_resolution),
        # safe_steps 4 (quinary image)
        (256 // 4, f"Output: Quinary image, Max diff tolerance: {256 // 4}"),
        default_required_match_rate,
    ),
    (
        "leres",
        "leres.onnx",
        lambda: LeresDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: LeresOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        default_tolerance,
        default_required_match_rate,
    ),
    (
        "leres[remove_near]",
        "leres.onnx",
        lambda: LeresDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: LeresOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", thr_a=30,
                                                   detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        # thr_a=30
        (int(255.0 * 30.0 / 100.0) + 1,
         f"Output: Filtered 30% near, Max diff tolerance: 255 * 30% + 1 = {int(255.0 * 30.0 / 100.0) + 1}"),
        default_required_match_rate,
    ),
    (
        "leres[remove_background]",
        "leres.onnx",
        lambda: LeresDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: LeresOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", thr_b=30,
                                                   detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        # thr_b=30
        (int(255.0 * 30.0 / 100.0) + 1,
         f"Output: Filtered 30% background, Max diff tolerance: 255 * 30% + 1 = {int(255.0 * 30.0 / 100.0) + 1}"),
        default_required_match_rate,
    ),
    (
        "lineart",
        "lineart.onnx",
        lambda: LineartDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: LineartOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        default_tolerance,
        default_required_match_rate,
    ),
    (
        "lineart_coarse",
        "lineart_coarse.onnx",
        lambda: LineartDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: LineartCoarseOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", coarse=True,
                                                   detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        default_tolerance,
        default_required_match_rate,
    ),
    (
        "lineart_anime",
        "lineart_anime.onnx",
        lambda: LineartAnimeDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: LineartAnimeOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        default_tolerance,
        default_required_match_rate,
    ),
    (
        "midas",
        "midas.onnx",
        lambda: MidasDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: MidasOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        default_tolerance,
        default_required_match_rate,
    ),
    (
        "midas[depth_and_normal]",
        "midas.onnx",
        lambda: MidasDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: MidasOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", depth_and_normal=True,
                                                   detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        (128, f"Output: Filtered 50%, Max diff tolerance: 128"),
        default_required_match_rate,
    ),
    (
        "zoe",
        "zoe_inference.onnx",
        lambda: ZoeDetector.from_pretrained("lllyasviel/Annotators"),
        # lambda: ZoeDetector.from_pretrained("Intel/zoedepth-nyu"),
        lambda model_dir, sess_options: ZoeOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        default_tolerance,
        default_required_match_rate,
    ),
    (
        "zoe[gamma_corrected]",
        "zoe_inference.onnx",
        lambda: ZoeDetector.from_pretrained("lllyasviel/Annotators"),
        # lambda: ZoeDetector.from_pretrained("Intel/zoedepth-nyu"),
        lambda model_dir, sess_options: ZoeOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", gamma_corrected=True,
                                                   detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        default_tolerance,
        default_required_match_rate,
    ),
    (
        "normalbae",
        "normalbae.onnx",
        lambda: NormalBaeDetector.from_pretrained("lllyasviel/Annotators"),
        lambda model_dir, sess_options: NormalBaeOnnx(model_dir=model_dir, sess_options=sess_options),
        lambda original_net, pil_img: original_net(pil_img, output_type="np", detect_resolution=default_resolution,
                                                   image_resolution=default_resolution),
        default_tolerance,
        default_required_match_rate,
    ),
    # 今後モデルを増やしたい場合は、ここにタプルを1行追加するだけで自動的にテスト対象になります。
]

@pytest.fixture(scope="module", params=[config[0] for config in MODEL_CONFIGS])
def get_model_data(request):
    """全テストを通して、ロードしたモデルを1度だけキャッシュするフィクスチャ"""
    cache = {}

    # テスト対象の設定マップを作成しておく
    config_map = {config[0]: config for config in MODEL_CONFIGS}
    model_name = request.param

    def _get_model():
        model_name = request.param
        (model_name, onnx_name, get_original, get_onnx_detector, run_detector,
         tolerance, required_match_rate) = config_map[model_name]

        # すでにロード済みならそれを返す
        if model_name in cache:
            return model_name, onnx_name, run_detector, tolerance, required_match_rate, cache[model_name]

        # 初めて要求された場合のみロードする
        if model_name not in config_map:
            raise ValueError(f"未知のモデル名です: {model_name}")

        # onnx_path = os.path.join(ONNX_DIR, f"{onnx_name}")
        # if not os.path.exists(onnx_path):
        #     return None  # ONNXがない場合はNoneを返す（テスト側でスキップさせる）

        # CPU
        sess_options = ort.SessionOptions()

        sess_options.intra_op_num_threads = 8
        # sess_options.inter_op_num_threads = 8
        # sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        # NPU
        # import qnn_ep_htp as qnn
        # sess_options = qnn.sess_options
        warnings.simplefilter("ignore")
        original_model = get_original()
        warnings.resetwarnings()
        cache[model_name] = {
            "original": original_model,
            "onnx": get_onnx_detector(model_dir=ONNX_DIR, sess_options=sess_options)
        }

        if tolerance[1] is not None:
            print(f"\n### NOTICE[{model_name}]: {tolerance[1]} ###", end="")
        return model_name, onnx_name, run_detector, tolerance, required_match_rate, cache[model_name]

    # テスト関数には関数（クロージャ）を渡す
    yield _get_model

    # テスト全体の終了後に一括解放
    for name in list(cache.keys()):
        del cache[name]["original"]
        del cache[name]["onnx"]
    cache.clear()
    import gc
    gc.collect()

# ==========================================
# 📦 共通ユーティリティ
# ==========================================
def get_test_images():
    """指定されたディレクトリから画像パスのリストを取得する"""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))
    return sorted(list(set(image_paths)))

def check_match_rate(original_output, onnx_output, image_path, model_name, tolerance, required_match_rate, id=""):
    # 検証
    assert original_output.shape == onnx_output.shape, f"形状が一致しません: Original {original_output.shape} vs ONNX {onnx_output.shape}"

    # For CPU(fp32 precision)
    # 輝度値のチェック
    # ピクセル値の誤差を許容するため assert_allclose を使用
    # atol (絶対誤差許容値) は実行環境(PyTorch、OnnxRuntime)の違いによる誤差を考慮して 1 に設定
    # np.testing.assert_allclose(
    #     onnx_output,
    #     original_output,
    #     rtol=0,
    #     atol=1,
    #     err_msg=f"画像 {os.path.basename(image_path)} の出力値が元の実装と一致しません。"
    # )

    # absolute_diff = np.abs(onnx_output - original_output)
    absolute_diff = np.abs(onnx_output.astype(np.int32) - original_output.astype(np.int32))
    is_perfect_match = absolute_diff == 0
    match_rate = np.mean(is_perfect_match)

    # print(onnx_output)
    # print(original_output)

    # path = image_path
    # result_img = original_output
    # file_path = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0], f"{model_name}_{id}_0.png")
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Image.fromarray(result_img).save(file_path)

    # result_img = onnx_output
    # file_path = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0], f"{model_name}_{id}_1.png")
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Image.fromarray(result_img).save(file_path)

    print(f"\n    match rate: {match_rate * 100:>8.3f}%, max diff: {np.max(absolute_diff):>3}", end=", ")

    assert match_rate >= required_match_rate and np.max(absolute_diff) <= tolerance[0], (
        f"\nモデル [{model_name}] - 画像 {os.path.basename(image_path)} の精度が基準未満です。\n"
        f"一致率: {match_rate * 100:.3f}% (目標: {required_match_rate * 100:.1f}%以上)\n"
        f"最大誤差: {np.max(absolute_diff):.4f}"
    )

# ==========================================
# 🧪 テストケース本体
# ==========================================

# 1. 精度検証テスト (画像×モデルの全組み合わせを自動生成)
@pytest.mark.parametrize("image_path", get_test_images())
def test_output_match(image_path, get_model_data):
    model_name, onnx_name, run_detector, tolerance, required_match_rate, models = get_model_data()

    # テスト対象のONNXが存在しない場合はスキップ (HEDがまだない時などのため)
    # onnx_path = os.path.join(ONNX_DIR, f"{onnx_name}")
    # if not os.path.exists(onnx_path):
    #     pytest.skip(f"ONNXファイルが見つからないためスキップします: {onnx_path}")

    if models is None:
        pytest.skip(f"モデルの初期化に失敗、またはONNXがありません: {model_name}")

    original_net = models["original"]
    onnx_detector = models["onnx"]

    # 画像の読み込み
    pil_img = Image.open(image_path).convert("RGB").resize((512, 512), Image.Resampling.BOX)

    # オリジナル推論
    original_output = run_detector(original_net, pil_img)

    # 自作ONNX推論 (統一I/O仕様に準拠)
    onnx_output = run_detector(onnx_detector, np.array(pil_img)[..., ::-1])

    if isinstance(original_output, tuple):
        check_match_rate(original_output[0], onnx_output[0], image_path, model_name, tolerance, required_match_rate, "main")
        check_match_rate(original_output[1], onnx_output[1], image_path, model_name, tolerance, required_match_rate, "sub")
    else:
        check_match_rate(original_output, onnx_output, image_path, model_name, tolerance, required_match_rate)


@pytest.mark.benchmark
# 2. 速度比較テスト (モデルごとに一括で計測)
def test_model_benchmark(get_model_data):
    model_name, onnx_name, run_detector, tolerance, required_match_rate, models = get_model_data()

    # onnx_path = os.path.join(ONNX_DIR, f"{model_name}_ctx.onnx")
    # onnx_path = os.path.join(ONNX_DIR, f"{onnx_name}")
    # if not os.path.exists(onnx_path):
    #     pytest.skip(f"ONNXファイルが見つからないためベンチマークをスキップします: {onnx_path}")

    image_paths = get_test_images()
    if not image_paths:
        pytest.skip("計測用の画像が見つかりません。")

    import time

    if models is None:
        pytest.skip(f"モデルの初期化に失敗、またはONNXがありません: {model_name}")

    original_net = models["original"]
    onnx_detector = models["onnx"]

    # ウォームアップ
    dummy_pil = Image.open(image_paths[0]).convert("RGB").resize((512, 512), Image.Resampling.BOX)
    _ = run_detector(original_net, dummy_pil)
    _ = run_detector(onnx_detector, dummy_pil)

    # オリジナル計測
    original_time = 0.0
    onnx_time = 0.0
    for path in image_paths:
        pil_img = Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.BOX)
        pil_img_bgr = np.array(pil_img)[..., ::-1].copy()

        start = time.perf_counter()
        result_img = run_detector(original_net, pil_img)
        original_time += time.perf_counter() - start

        # file_path = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0], f"{model_name}_0.png")
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Image.fromarray(result_img).save(file_path)

        start = time.perf_counter()
        result_img = run_detector(onnx_detector, pil_img_bgr)
        onnx_time += time.perf_counter() - start

        # file_path = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0], f"{model_name}_1.png")
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Image.fromarray(result_img).save(file_path)

    # レポート
    num_images = len(image_paths)
    avg_orig = (original_time / num_images) * 1000
    avg_onnx = (onnx_time / num_images) * 1000

    print(f"\n================ BENCHMARK REPORT: [{model_name.upper()}] ({num_images} images) ================")
    print(f"Original controlnet_aux Avg Time: {avg_orig:.2f} ms / image")
    print(f"ONNX Model Avg Time:            {avg_onnx:.2f} ms / image")
    print(f"ONNX is {avg_orig / avg_onnx:.2f}x faster than Original!")
    print("=========================================================================================")
    assert True

