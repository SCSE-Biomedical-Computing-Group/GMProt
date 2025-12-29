import zipfile
from pathlib import Path

def extract_all_zips(zip_folder: str | Path, out_folder: str | Path):
    """
    Extract all .zip files from `zip_folder` into `out_folder`.
    
    - zip_folder: directory containing .zip files
    - out_folder: destination directory where everything is extracted
    - Creates out_folder if it does not exist
    - Each zip file is extracted into a subfolder named after the zip file
    
    Example:
        zip1.zip   -> out_folder/zip1/
        zip2.zip   -> out_folder/zip2/
    """
    zip_folder = Path(zip_folder)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    zip_files = list(zip_folder.glob("*.zip"))

    if not zip_files:
        print(f"No zip files found in {zip_folder}")
        return

    for zip_path in zip_files:
        extract_subdir = out_folder / zip_path.stem
        extract_subdir.mkdir(exist_ok=True)

        print(f"Extracting {zip_path.name} → {extract_subdir}")

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_subdir)

    print(f"Extraction completed. Output stored in: {out_folder}")

if __name__ == "__main__":
    extract_all_zips(
    zip_folder="/data/prem001/PGAT-ABPp/code/data/alphafold_pdb/all_outputs",
    out_folder="/data/prem001/PGAT-ABPp/code/data/alphafold_pdb/raw_pdb"
)
