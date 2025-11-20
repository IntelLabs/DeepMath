# combine multiple files with the same prefix into one file
import glob
import re


def main():
    files = glob.glob("*.jsonl")
    groups = {}
    pattern = re.compile(r"^(.*)_\d{3}_\d{3}\.jsonl$")
    for file in files:
        m = pattern.match(file)
        if m:
            prefix = m.group(1)
            groups.setdefault(prefix, []).append(file)
    for prefix, group_files in groups.items():
        group_files.sort()
        output_file = prefix + ".jsonl"
        if output_file in files:
            print(f"Skipping {output_file}: already exists.")
            continue
        print("Merging files:", group_files)
        print("Output file:", output_file)
        with open(output_file, "w") as outfile:
            for file in group_files:
                with open(file) as infile:
                    for line in infile:
                        outfile.write(line)


if __name__ == "__main__":
    main()
