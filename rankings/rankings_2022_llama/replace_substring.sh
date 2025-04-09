#!/bin/bash

# Declare an array of old and new substrings
declare -A SUBSTRING_MAP
SUBSTRING_MAP["role_narrative_chainofth"]="RNC"
SUBSTRING_MAP["norole_narrative_chainofth"]="NC"
SUBSTRING_MAP["role_nonarrative_chainofth"]="RC"
SUBSTRING_MAP["norole_nonarrative_chainofth"]="C"
SUBSTRING_MAP["gen_narr_trec_"]=""
SUBSTRING_MAP["original"]="orig"
SUBSTRING_MAP["_title_cleanhtml"]=""
SUBSTRING_MAP["_question"]=""
SUBSTRING_MAP["MiniLM"]="miniLM"
SUBSTRING_MAP["_no"]="_"


#Loop through all files in the current directory
for file in *; do
  # Only process files
  if [[ -f "$file" ]]; then
    new_name="$file"
    
    # Apply all substring replacements
    for OLD_SUBSTRING in "${!SUBSTRING_MAP[@]}"; do
      NEW_SUBSTRING="${SUBSTRING_MAP[$OLD_SUBSTRING]}"
      new_name="${new_name//$OLD_SUBSTRING/$NEW_SUBSTRING}"
    done

    # Rename if the name has changed
    if [[ "$new_name" != "$file" ]]; then
      echo "Renaming '$file' to '$new_name'"
      mv "$file" "$new_name"
    fi
  fi
done

