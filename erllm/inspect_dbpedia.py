import re
import sqlite3
from erllm import DBFILE_PATH, RUNS_FOLDER_PATH
from erllm.llm_matcher.evalrun import read_run_raw

# Define regex pattern to match entity descriptions
entity_pattern = re.compile(r"Entity \d: '(.+?)'")

runfile = (
    RUNS_FOLDER_PATH
    / "35_base"
    / "full_dbpedia"
    / "dbpedia10k-general_complex_force-gpt_3.5_turbo_instruct-1max_token_0.json"
)

completions = read_run_raw(runfile)
match_completions = [
    completion for completion in completions.values() if completion.truth
]
report_prompt_strings = [completion.prompt_string for completion in match_completions]
# Generate report
entity_strings = []
print("Report of prompt strings where truth is True:")
for prompt_string in report_prompt_strings:
    matches = entity_pattern.findall(prompt_string)
    e0 = matches[0]
    e1 = matches[1]
    entity_strings.append((e0, e1))
print(f"# matches: {len(entity_strings)}")
# count the tuples in entity_strings where both elements are equal
count = sum(e0 == e1 for e0, e1 in entity_strings)
print(f"# equal entities: {count} ({count/len(entity_strings)*100:.2f}%)")

conn = sqlite3.connect(DBFILE_PATH)
cursor = conn.cursor()
# Execute the query
cursor.execute(
    "SELECT \
    (SELECT COUNT(*) FROM dbpedia_matches) AS total_rows, \
    (SELECT COUNT(DISTINCT id0) FROM dbpedia_matches) AS distinct_id0, \
    (SELECT COUNT(DISTINCT id1) FROM dbpedia_matches) AS distinct_id1;"
)

results = cursor.fetchone()
print("Total Rows:", results[0])
print("Distinct id0:", results[1])
print("Distinct id1:", results[2])


cursor.execute("SELECT COUNT(DISTINCT id) FROM dbpedia0;")
distinct_id_count = cursor.fetchone()[0]
print("Number of distinct id values in dbpedia0:", distinct_id_count)


cursor.execute("SELECT COUNT(DISTINCT id) FROM dbpedia1;")
distinct_id_count = cursor.fetchone()[0]
print("Number of distinct id values in dbpedia1:", distinct_id_count)
