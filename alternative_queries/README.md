The jsonl files only contain the query id and the variants of the queries.

The xml files provide more information. In particular, both the original query and the variants are included. In order to preserve the notation of the original datasets, we used the following convention:
* 2020:
    - The field 'title' contains the original query. 
    - The field 'description' contains a variant.
    - The field 'narrative' contains the original/generated narrative.
* 2021:
    - The field 'query' contains the original query. 
    - The field 'description contains a variant.
    - The field 'narrative' contains the original/generated narrative.
* 2022:
    - The field 'query' contains the original query. 
    - The field 'question' contains a variant.
    - The field 'background' contains the original/generated narrative.
* CLEF:
    - The field 'originaltitle' contains the original query. 
    - The field 'title' contains a variant.
    - The field 'narrative' contains an empty string/a generated narrative.
    