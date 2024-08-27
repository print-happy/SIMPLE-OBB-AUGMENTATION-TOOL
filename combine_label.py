def merge_txt_files(file1_path, file2_path, output_file_path):

    with open(file1_path, 'r') as file1:
        file1_lines = file1.readlines()

    with open(file2_path, 'r') as file2:
        file2_lines = file2.readlines()

    merged_lines = file1_lines + file2_lines

    with open(output_file_path, 'w') as output_file:
        output_file.writelines(merged_lines)

    print(f"Successfully combined to {output_file_path}")

file1_path = ''
file2_path = ''
output_file_path = ''

merge_txt_files(file1_path, file2_path, output_file_path)
