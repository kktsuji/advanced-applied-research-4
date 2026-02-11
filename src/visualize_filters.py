"""
フィルタの周波数応答とノイズ特性の可視化

このスクリプトは、レポートのセクション2で説明されている各種フィルタとノイズの
周波数特性を可視化します。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["font.size"] = 10


def create_output_dir(dirname="out/filters"):
    """出力ディレクトリを作成"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


def plot_box_filter_frequency_response(sizes=[3, 5, 7, 9], output_dir="out/filters"):
    """ボックスフィルタの周波数応答をプロット"""
    plt.figure(figsize=(10, 6))

    for size in sizes:
        # 1次元ボックスフィルタ
        h = np.ones(size) / size

        # 周波数応答を計算
        w, H = signal.freqz(h, worN=2048)

        # dBスケールでプロット
        plt.plot(
            w / np.pi,
            20 * np.log10(np.abs(H) + 1e-10),
            label=f"Size {size}x{size}",
            linewidth=2,
        )

    plt.title("Box Filter Frequency Response", fontsize=14, fontweight="bold")
    plt.xlabel("Normalized Frequency (×π rad/sample)", fontsize=12)
    plt.ylabel("Magnitude (dB)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim([-60, 5])
    plt.tight_layout()

    filepath = os.path.join(output_dir, "box_filter_frequency_response.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.close()


def plot_gaussian_filter_frequency_response(
    sigmas=[0.5, 1.0, 2.0, 3.0], output_dir="out/filters"
):
    """ガウシアンフィルタの周波数応答をプロット"""
    plt.figure(figsize=(10, 6))

    for sigma in sigmas:
        # ガウシアンカーネルを生成（十分なサイズ）
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
        center = size // 2
        x = np.arange(size) - center

        # 1次元ガウス関数
        h = np.exp(-(x**2) / (2 * sigma**2))
        h = h / np.sum(h)  # 正規化

        # 周波数応答を計算
        w, H = signal.freqz(h, worN=2048)

        # dBスケールでプロット
        plt.plot(
            w / np.pi,
            20 * np.log10(np.abs(H) + 1e-10),
            label=f"σ = {sigma}",
            linewidth=2,
        )

    plt.title("Gaussian Filter Frequency Response", fontsize=14, fontweight="bold")
    plt.xlabel("Normalized Frequency (×π rad/sample)", fontsize=12)
    plt.ylabel("Magnitude (dB)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim([-60, 5])
    plt.tight_layout()

    filepath = os.path.join(output_dir, "gaussian_filter_frequency_response.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.close()


def plot_filter_comparison(output_dir="out/filters"):
    """ボックス、ガウシアン、理想ローパスフィルタの比較"""
    plt.figure(figsize=(12, 6))

    # ボックスフィルタ (5x5)
    h_box = np.ones(5) / 5
    w, H_box = signal.freqz(h_box, worN=2048)

    # ガウシアンフィルタ (σ=1.0)
    sigma = 1.0
    size = 11
    center = size // 2
    x = np.arange(size) - center
    h_gauss = np.exp(-(x**2) / (2 * sigma**2))
    h_gauss = h_gauss / np.sum(h_gauss)
    _, H_gauss = signal.freqz(h_gauss, worN=2048)

    # 理想ローパスフィルタ（参考）
    cutoff = 0.3  # 正規化カットオフ周波数
    H_ideal = np.where(w / np.pi <= cutoff, 1.0, 0.0)

    # プロット
    plt.plot(
        w / np.pi,
        20 * np.log10(np.abs(H_box) + 1e-10),
        label="Box Filter (5x5)",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        w / np.pi,
        20 * np.log10(np.abs(H_gauss) + 1e-10),
        label="Gaussian Filter (σ=1.0)",
        linewidth=2,
    )
    plt.plot(
        w / np.pi,
        20 * np.log10(H_ideal + 1e-10),
        label="Ideal Lowpass (cutoff=0.3π)",
        linewidth=2,
        linestyle=":",
    )

    plt.title(
        "Comparison of Filter Frequency Responses", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Normalized Frequency (×π rad/sample)", fontsize=12)
    plt.ylabel("Magnitude (dB)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim([-60, 5])
    plt.tight_layout()

    filepath = os.path.join(output_dir, "filter_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.close()


def visualize_filter_kernels_2d(output_dir="out/filters"):
    """2次元フィルタカーネルの可視化"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ボックスフィルタ (5x5)
    box_kernel = np.ones((5, 5)) / 25

    # ガウシアンフィルタ (5x5, σ=1.0)
    sigma = 1.0
    size = 5
    center = size // 2
    x = np.arange(size) - center
    y = np.arange(size) - center
    X, Y = np.meshgrid(x, y)
    gauss_kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    gauss_kernel = gauss_kernel / np.sum(gauss_kernel)

    # イプシロンフィルタ（例：中心からの距離に基づく重み）
    # 注：イプシロンフィルタは信号依存なので、これは一例
    epsilon = 10
    distances = np.sqrt(X**2 + Y**2) * 20  # 仮の輝度差
    epsilon_kernel = 1 / (1 + (distances / epsilon) ** 2)
    epsilon_kernel = epsilon_kernel / np.sum(epsilon_kernel)

    # ボックスフィルタ
    im0 = axes[0].imshow(box_kernel, cmap="hot", interpolation="nearest")
    axes[0].set_title("Box Filter (5x5)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    # ガウシアンフィルタ
    im1 = axes[1].imshow(gauss_kernel, cmap="hot", interpolation="nearest")
    axes[1].set_title("Gaussian Filter (5x5, σ=1.0)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1])

    # イプシロンフィルタ（例）
    im2 = axes[2].imshow(epsilon_kernel, cmap="hot", interpolation="nearest")
    axes[2].set_title("Epsilon Filter (example, ε=10)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    filepath = os.path.join(output_dir, "filter_kernels_2d.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.close()


def visualize_epsilon_filter_weight_function(output_dir="out/filters"):
    """イプシロンフィルタの重み関数を可視化"""
    plt.figure(figsize=(10, 6))

    # 輝度差の範囲
    x = np.linspace(-100, 100, 1000)

    # 異なるε値での重み関数
    epsilons = [5, 10, 20, 40]

    for eps in epsilons:
        w = 1 / (1 + (np.abs(x) / eps) ** 2)
        plt.plot(x, w, label=f"ε = {eps}", linewidth=2)

    plt.title("Epsilon Filter Weight Function", fontsize=14, fontweight="bold")
    plt.xlabel("Intensity Difference |I[i,j] - I[m,n]|", fontsize=12)
    plt.ylabel("Weight w(x)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim([-100, 100])
    plt.ylim([0, 1.05])

    # エッジ領域と平坦領域を示す
    plt.axvspan(-100, -30, alpha=0.2, color="red", label="Edge region")
    plt.axvspan(30, 100, alpha=0.2, color="red")
    plt.axvspan(-20, 20, alpha=0.2, color="green", label="Flat region")

    plt.tight_layout()

    filepath = os.path.join(output_dir, "epsilon_filter_weight_function.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.close()


def generate_noise_samples(output_dir="out/filters"):
    """各種ノイズサンプルの生成と可視化"""
    # テスト画像の作成（グラデーション + 矩形）
    size = 256
    image = np.zeros((size, size))

    # グラデーション
    for i in range(size):
        image[i, :] = i / size * 255

    # 矩形を追加
    image[80:180, 80:180] = 200

    # 各種ノイズを追加
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 元画像
    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # ガウシアンノイズ
    gaussian_noise = np.random.normal(0, 25, image.shape)
    noisy_gaussian = np.clip(image + gaussian_noise, 0, 255)
    axes[0, 1].imshow(noisy_gaussian, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Gaussian Noise (σ=25)", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    # ショットノイズ（ポアソンノイズ近似）
    # 信号依存性を持つノイズ
    noisy_poisson = image + np.random.randn(*image.shape) * np.sqrt(
        np.maximum(image, 1)
    )
    noisy_poisson = np.clip(noisy_poisson, 0, 255)
    axes[0, 2].imshow(noisy_poisson, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Shot Noise (Poisson-like)", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    # 塩胡椒ノイズ
    noisy_sp = image.copy()
    prob = 0.05
    # 塩（白）
    salt_coords = np.random.random(image.shape) < prob / 2
    noisy_sp[salt_coords] = 255
    # 胡椒（黒）
    pepper_coords = np.random.random(image.shape) < prob / 2
    noisy_sp[pepper_coords] = 0
    axes[1, 0].imshow(noisy_sp, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Salt & Pepper Noise (p=0.05)", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    # 固定パターンノイズ
    # 低周波のパターンノイズを生成
    x = np.linspace(0, 4 * np.pi, size)
    y = np.linspace(0, 4 * np.pi, size)
    X, Y = np.meshgrid(x, y)
    fixed_pattern = 30 * np.sin(X) * np.cos(Y)
    noisy_fpn = np.clip(image + fixed_pattern, 0, 255)
    axes[1, 1].imshow(noisy_fpn, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Fixed Pattern Noise", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    # 複合ノイズ（ガウシアン + 塩胡椒）
    noisy_combined = image + np.random.normal(0, 15, image.shape)
    salt_coords = np.random.random(image.shape) < 0.02
    noisy_combined[salt_coords] = 255
    pepper_coords = np.random.random(image.shape) < 0.02
    noisy_combined[pepper_coords] = 0
    noisy_combined = np.clip(noisy_combined, 0, 255)
    axes[1, 2].imshow(noisy_combined, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title("Combined Noise", fontsize=12, fontweight="bold")
    axes[1, 2].axis("off")

    plt.tight_layout()

    filepath = os.path.join(output_dir, "noise_samples.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.close()


def visualize_noise_frequency_spectrum(output_dir="out/filters"):
    """ノイズの周波数スペクトルを可視化"""
    size = 256

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # ガウシアンノイズ（ホワイトノイズ）
    gaussian_noise = np.random.normal(0, 25, (size, size))
    fft_gaussian = np.fft.fft2(gaussian_noise)
    fft_gaussian_shifted = np.fft.fftshift(fft_gaussian)
    magnitude_gaussian = np.abs(fft_gaussian_shifted)
    axes[0, 0].imshow(np.log1p(magnitude_gaussian), cmap="hot")
    axes[0, 0].set_title("Gaussian Noise Spectrum", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # 塩胡椒ノイズ
    sp_noise = np.zeros((size, size))
    salt_coords = np.random.random((size, size)) < 0.025
    sp_noise[salt_coords] = 255
    pepper_coords = np.random.random((size, size)) < 0.025
    sp_noise[pepper_coords] = -255
    fft_sp = np.fft.fft2(sp_noise)
    fft_sp_shifted = np.fft.fftshift(fft_sp)
    magnitude_sp = np.abs(fft_sp_shifted)
    axes[0, 1].imshow(np.log1p(magnitude_sp), cmap="hot")
    axes[0, 1].set_title("Salt & Pepper Noise Spectrum", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    # 固定パターンノイズ（低周波）
    x = np.linspace(0, 4 * np.pi, size)
    y = np.linspace(0, 4 * np.pi, size)
    X, Y = np.meshgrid(x, y)
    fixed_pattern = 30 * np.sin(X) * np.cos(Y)
    fft_fpn = np.fft.fft2(fixed_pattern)
    fft_fpn_shifted = np.fft.fftshift(fft_fpn)
    magnitude_fpn = np.abs(fft_fpn_shifted)
    axes[0, 2].imshow(np.log1p(magnitude_fpn), cmap="hot")
    axes[0, 2].set_title("Fixed Pattern Noise Spectrum", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    # 周波数プロファイル（中心からの放射方向）
    center = size // 2

    # ガウシアンノイズの放射プロファイル
    y_profile_gauss = magnitude_gaussian[center, center:]
    axes[1, 0].plot(y_profile_gauss)
    axes[1, 0].set_title("Gaussian Noise Radial Profile", fontsize=10)
    axes[1, 0].set_xlabel("Frequency")
    axes[1, 0].set_ylabel("Magnitude")
    axes[1, 0].grid(True, alpha=0.3)

    # 塩胡椒ノイズの放射プロファイル
    y_profile_sp = magnitude_sp[center, center:]
    axes[1, 1].plot(y_profile_sp)
    axes[1, 1].set_title("Salt & Pepper Radial Profile", fontsize=10)
    axes[1, 1].set_xlabel("Frequency")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].grid(True, alpha=0.3)

    # 固定パターンノイズの放射プロファイル
    y_profile_fpn = magnitude_fpn[center, center:]
    axes[1, 2].plot(y_profile_fpn)
    axes[1, 2].set_title("Fixed Pattern Radial Profile", fontsize=10)
    axes[1, 2].set_xlabel("Frequency")
    axes[1, 2].set_ylabel("Magnitude")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = os.path.join(output_dir, "noise_frequency_spectrum.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.close()


def compare_filter_results_on_noisy_image(output_dir="out/filters"):
    """エッジを含むテスト画像でのフィルタリング結果比較"""
    # テスト画像の作成（エッジとテクスチャを含む）
    size = 256
    image = np.zeros((size, size))

    # 複数の矩形でエッジを作成
    image[50:100, 50:200] = 100
    image[120:200, 80:180] = 200
    image[150:220, 150:230] = 150

    # ガウシアンノイズを追加
    noisy_image = image + np.random.normal(0, 20, image.shape)
    noisy_image = np.clip(noisy_image, 0, 255)

    # 各フィルタでフィルタリング
    # ボックスフィルタ
    box_filtered = ndimage.uniform_filter(noisy_image, size=5)

    # ガウシアンフィルタ
    gaussian_filtered = ndimage.gaussian_filter(noisy_image, sigma=1.5)

    # メディアンフィルタ（イプシロンフィルタの代わり）
    median_filtered = ndimage.median_filter(noisy_image, size=5)

    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(noisy_image, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Noisy Image (σ=20)", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(box_filtered, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Box Filter (5x5)", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(gaussian_filtered, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Gaussian Filter (σ=1.5)", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(median_filtered, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Median Filter (5x5)", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    # エッジ保存性の定量評価（エッジ強度）
    edge_original = np.abs(ndimage.sobel(image))
    edge_box = np.abs(ndimage.sobel(box_filtered))
    edge_gaussian = np.abs(ndimage.sobel(gaussian_filtered))
    edge_median = np.abs(ndimage.sobel(median_filtered))

    edge_preservation = [
        ("Original", np.mean(edge_original)),
        ("Box", np.mean(edge_box)),
        ("Gaussian", np.mean(edge_gaussian)),
        ("Median", np.mean(edge_median)),
    ]

    axes[1, 2].bar([x[0] for x in edge_preservation], [x[1] for x in edge_preservation])
    axes[1, 2].set_title(
        "Edge Preservation (avg magnitude)", fontsize=12, fontweight="bold"
    )
    axes[1, 2].set_ylabel("Average Edge Magnitude")
    axes[1, 2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    filepath = os.path.join(output_dir, "filter_comparison_on_image.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.close()


def main():
    """メイン処理：全ての可視化を実行"""
    print("=" * 60)
    print("フィルタとノイズの周波数特性可視化スクリプト")
    print("=" * 60)

    # 出力ディレクトリを作成
    output_dir = create_output_dir()
    print(f"\n出力ディレクトリ: {output_dir}\n")

    print("1. ボックスフィルタの周波数応答を生成中...")
    plot_box_filter_frequency_response(output_dir=output_dir)

    print("2. ガウシアンフィルタの周波数応答を生成中...")
    plot_gaussian_filter_frequency_response(output_dir=output_dir)

    print("3. フィルタ比較グラフを生成中...")
    plot_filter_comparison(output_dir=output_dir)

    print("4. 2Dフィルタカーネルを可視化中...")
    visualize_filter_kernels_2d(output_dir=output_dir)

    print("5. イプシロンフィルタの重み関数を可視化中...")
    visualize_epsilon_filter_weight_function(output_dir=output_dir)

    print("6. ノイズサンプルを生成中...")
    generate_noise_samples(output_dir=output_dir)

    print("7. ノイズの周波数スペクトルを可視化中...")
    visualize_noise_frequency_spectrum(output_dir=output_dir)

    print("8. テスト画像でのフィルタ比較を生成中...")
    compare_filter_results_on_noisy_image(output_dir=output_dir)

    print("\n" + "=" * 60)
    print("全ての可視化が完了しました！")
    print(f"生成されたファイルは {output_dir} ディレクトリにあります。")
    print("=" * 60)


if __name__ == "__main__":
    main()
