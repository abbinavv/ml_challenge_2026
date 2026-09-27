"""Build <team>_submission.zip in the organisers' structure from an output folder.

Usage: python src/make_package.py <output_dir> [team_name]
Zip: output/{matching_results.tsv,candidate_pairs.tsv}, code/business_entity_resolution/{src,aws,
README.md,requirements.txt}, Documentation_template.md. Refuses to package if the official
checker (utils/validate_submission.py) fails or if a file looks like it carries credentials.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

CHECKER = os.environ.get("BER_CHECKER", "/Users/main/Downloads/student_resource/utils/validate_submission.py")
TEST_DIR = os.environ.get("BER_TEST_DIR", "data/dataset/test")
# patterns assembled from parts so this file does not match itself
SECRET = re.compile("AK" + "IA[0-9A-Z]{16}|" + "aws_secret_" + "access_key|" + "Secret" + "AccessKey")


def main():
    out_dir = sys.argv[1]
    team = sys.argv[2] if len(sys.argv) > 2 else "Dronaut"
    m, c = os.path.join(out_dir, "matching_results.tsv"), os.path.join(out_dir, "candidate_pairs.tsv")
    r = subprocess.run([sys.executable, CHECKER, "-m", m, "-c", c, "-t", TEST_DIR, "--check-ids"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("checker failed:\n" + r.stdout[-2000:])
    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, f"{team}_submission")
    code = os.path.join(root, "code", "business_entity_resolution")
    os.makedirs(os.path.join(root, "output")); os.makedirs(code)
    shutil.copy(m, os.path.join(root, "output")); shutil.copy(c, os.path.join(root, "output"))
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree("src", os.path.join(code, "src"), ignore=ign)
    shutil.copytree("aws", os.path.join(code, "aws"), ignore=ign)
    for f in ("README.md", "requirements.txt"):
        shutil.copy(f, code)
    shutil.copy("Documentation_template.md", root)
    for d, _, fs in os.walk(code):
        for f in fs:
            if SECRET.search(open(os.path.join(d, f), errors="ignore").read()):
                sys.exit(f"possible credential in {f}; not packaging")
    zpath = f"{team}_submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, fs in os.walk(root):
            for f in fs:
                full = os.path.join(d, f)
                z.write(full, os.path.relpath(full, root))
    shutil.rmtree(tmp)
    print(f"wrote {zpath} ({os.path.getsize(zpath) / 1e6:.0f} MB) from {out_dir}")


if __name__ == "__main__":
    main()
