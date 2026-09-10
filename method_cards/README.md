# Method cards

Private source Markdown stays outside this repository. Inventory records only source locators,
hashes, headings, section ranges, and exclusion status; it never copies raw source text.

Draft extraction accepts explicit fenced blocks whose info string is `method-card`. The block must
contain the required `MethodCard` fields in YAML and already be paraphrased. Its evidence level must
match the inventoried source type. Extraction always forces `draft` status, and only `alpha-lab
cards approve` can record an approving reviewer and timestamp.
