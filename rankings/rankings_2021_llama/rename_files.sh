#!/bin/bash

# Declare an array of old and new prefixes
declare -A PREFIX_MAP
PREFIX_MAP["llama_rerank_bm25-cross-encoder_cross-encoder-ms-marco-MiniLM-L-12-v2"]="llama_C4-2022_MiniLM-L-12-v2"
PREFIX_MAP["llama_all_res_C4-2022_bm25"]="llama_C4-2022_bm25"

# Loop over each prefix pair
for OLD_PREFIX in "${!PREFIX_MAP[@]}"; do
  NEW_PREFIX="${PREFIX_MAP[$OLD_PREFIX]}"

  # Process matching files
  for file in "$OLD_PREFIX"*; do
    if [[ -f "$file" ]]; then
      new_name="${file/#$OLD_PREFIX/$NEW_PREFIX}"
      echo "Renaming '$file' to '$new_name'"
      mv "$file" "$new_name"
    fi
  done
done
