import httpx
import os


URL = "http://images.cocodataset.org/val2017"

image_names = [
    "000000000285.jpg",
    "000000000872.jpg",
    "000000017029.jpg",
    "000000031749.jpg",
    "000000039670.jpg",
    "000000050896.jpg",
    "000000080666.jpg",
    "000000088848.jpg",
    "000000094852.jpg",
    "000000149406.jpg",
    "000000173799.jpg",
    "000000177065.jpg",
    "000000198510.jpg",
    "000000204186.jpg",
    "000000229858.jpg",
    "000000239274.jpg",
    "000000278353.jpg",
    "000000286908.jpg",
    "000000291664.jpg",
    "000000325031.jpg",
    "000000338191.jpg",
    "000000338219.jpg",
    "000000369812.jpg",
    "000000419098.jpg",
    "000000427256.jpg",
    "000000429718.jpg",
    "000000442306.jpg",
    "000000460967.jpg",
    "000000517832.jpg",
    "000000525286.jpg",
    "000000562229.jpg",
    "000000577932.jpg",
]


def download_files(file_list, output_dir="test_images"):
    os.makedirs(output_dir, exist_ok=True)

    with httpx.Client(timeout=None) as client:
        for filename in file_list:
            url = f"{URL}/{filename}"
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                print(f"SKIPPED: {filename} already exists.")
                continue

            try:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with open(output_path, "wb") as f:
                        for chunk in response.iter_bytes():
                            f.write(chunk)
                print(f"✅ Downloaded: {output_path}")
            except httpx.HTTPError as e:
                print(f"❌ Failed to download {url}: {e}")


if __name__ == "__main__":
    dir_path = f"{os.path.dirname(__file__)}/test_images"
    download_files(image_names, dir_path)
