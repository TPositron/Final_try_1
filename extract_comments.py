import os
import ast

def get_file_docstring(filepath):
    """Extract the top docstring from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=filepath)
        docstring = ast.get_docstring(tree)
        return docstring.strip() if docstring else None
    except (UnicodeDecodeError, SyntaxError, PermissionError) as e:
        print(f"[Error] Could not read {filepath}: {str(e)}")
        return None

def scan_python_files(root_dir):
    """Find all .py files in root_dir and subdirectories."""
    python_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    return python_files

def save_docstrings_to_txt(docstrings, output_file="docstrings.txt"):
    """Save extracted docstrings to a text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for filepath, doc in docstrings.items():
            if doc:  # Only write files with docstrings
                f.write(f"───── {filepath} ─────\n")
                f.write(doc + "\n\n")

def main():
    target_folder = r"C:\Users\tarik\Desktop\Image_analysis\src"
    output_file = "extracted_comments.txt"
    
    print(f"Scanning Python files in: {target_folder}")
    python_files = scan_python_files(target_folder)
    docstrings = {fp: get_file_docstring(fp) for fp in python_files}
    
    save_docstrings_to_txt(docstrings, output_file)
    
    total_files = len(python_files)
    files_with_docs = sum(1 for doc in docstrings.values() if doc)
    print(f"Processed {total_files} Python files.")
    print(f"Found docstrings in {files_with_docs} files.")
    print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    main()