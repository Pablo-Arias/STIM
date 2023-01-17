import numpy as np
import bibtexparser
import os
import glob
from difflib import SequenceMatcher


def write_entry_note(entry, destination, add_rating, add_last_abstract_sentence, tags):
    if add_rating:
        entry["rating"] = -1
    
    file_name = destination + entry["ID"] + ".md"
    
    f = open( file_name, "w")
    f.write("---\n")
    for key in entry.keys():
        to_add = str(entry[key])
        to_add = to_add.replace("{", "").replace("}", "")
        if key== "title":
            to_add = "'''"+to_add+"'''"
        f.write(key + " : "+ to_add + "\n") 
    f.write("\n")
    f.write("---\n")

    if len(tags)> 0:
        for tag in tags:
            f.write("#" + tag +" ")
        f.write("\n")
    
    #title
    title = entry["title"].replace("{", "").replace("}", "")        
    f.write("# " + title + "\n" )

    #authors
    authors = entry["author"]
    f.write(authors + "\n\n")
    
    if "url" in entry:
        f.write("[Access]("+entry["url"]+")\n\n")

    elif "doi" in entry:
        doi = entry["doi"]
        #doi = doi.replace("/", "%")
        f.write("[Access](https://scholar.google.fr/scholar?q=" + doi + ")\n")
        f.write("\n" )
        f.write("\n" )

    if "abstract" in entry:
        f.write("## Abstract \n") 
        f.write(entry["abstract"] + "\n" )
        f.write("\n" )

    f.write("## Extract \n" )
    if "abstract" in entry and add_last_abstract_sentence:
        abstract = entry["abstract"].split(".")
        abstract = [item for item in abstract if "reserved" not in item]
        abstract = [item for item in abstract if "textcopyright" not in item]
        abstract = [item for item in abstract if len(item) > 4]

        if len(abstract)>2:
            last = "Last sentence  : " + abstract[-1]
            f.write(last + "\n" )

    f.write("\n" )
    f.write("\n" )        

    f.write("## Notes \n" )
    f.write("\n" )        
    f.write("\n" )  
    f.close()

    return file_name

def generate_notes_from_bib(bib_file, destination="/Users/arias/Documents/p-vault/p-brain/Pro/generated/", add_rating=True, add_last_abstract_sentence=True, tags=[]):
    with open(bib_file) as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    #create a note for each entry in the databae
    for entry in bib_database.entries:
        write_entry_note(entry, destination, add_rating, add_last_abstract_sentence, tags)



def merge_notes(one_note_notes_folder, bibtex_file, target_merge_folder, add_rating=True, add_last_abstract_sentence=True, tags=[]):
    os.mkdir(target_merge_folder)

    #get bib tex information
    with open(bibtex_file) as bib_file:
        bib_database = bibtexparser.load(bib_file)

    for file in glob.glob(one_note_notes_folder+"*.md"):
        title = file.split("/")[-1]
        found=False
        for entry in bib_database.entries:
            bib_title = entry["title"].replace("{", "").replace("}", "")
            if SequenceMatcher(a=title,b=bib_title).ratio()>0.7:
                new_note = write_entry_note(entry
                    , destination=target_merge_folder
                    , add_rating=add_rating
                    , add_last_abstract_sentence=add_last_abstract_sentence
                    , tags = tags
                    )

                # Reading data from file1
                new_note = open( new_note, "a")
                one_note  = open( file, "r")
                data = one_note.read()
                new_note.write(data)
                found=True
                print("Merging the two following")
                print(title)
                print(bib_title)                
                print("")
                break


        if not found:
            print("not found")
            print(title)
            print(bib_title)
            print("")            

def generate_notes_from_title(titles, bibtex_file, target_folder, add_rating=True, add_last_abstract_sentence=True, tags=[]):
    os.mkdir(target_folder)

    #get bib tex information
    with open(bibtex_file) as bib_file:
        bib_database = bibtexparser.load(bib_file)

    for title in titles:
        found=False
        for entry in bib_database.entries:
            bib_title = entry["title"].replace("{", "").replace("}", "")
            if SequenceMatcher(a=title,b=bib_title).ratio()>0.7:
                new_note = write_entry_note(entry
                    , destination=target_folder
                    , add_rating=add_rating
                    , add_last_abstract_sentence=add_last_abstract_sentence
                    , tags = tags
                    )

                print("Generating the following note")
                print(title)
                print("")
                break


        if not found:
            print("not found")
            print(title)
            print(bib_title)
            print("")            
            