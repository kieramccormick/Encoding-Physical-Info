# Code used for generating the LLM-generated summaries with context using the updated prompt (for astrophysical X-ray sources)

import random
import json
import openai
from openai import OpenAI
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pyvo as vo
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import pandas as pd
import h5py
import argparse
import os
import gzip
import chardet
import ijson
import seaborn as sns
import re

# Input the file that contains the bibliographic data and the association with Chandra obsids
file_path = 'ChandraBib_OA.jsonl.gz'

# This file contains the targets names and the target coordinates for each obsid. This is from the Chandra Chaser
df = pd.read_csv('target_obsid_coordinates.csv')

# This file contains the successfully processed obsids (i.e., those that meet the S/N and theta requierements and have sources)
df_obsids = pd.read_csv('processed_obsids_sig_gt_5_theta_lt_5.csv',names=['obsid'], skiprows=1)

# This file contains all the brightest and most significant ObsIDs
df_bright_obsids = pd.read_csv('bright_sources.csv')

# CSC 2.1 TAP service
tap = vo.dal.TAPService('http://cda.cfa.harvard.edu/csc21tap/') # For CSC 2.1

# Replace with your own API key
client = OpenAI(api_key='xxx')

# Model names
generation_model = "gpt-4o-mini"  # or the model you are using for text generation
embedding_model = "text-embedding-ada-002"  # Use this for generating embeddings

def stream_search_obsid(file_path, target_obsid):
    with gzip.open(file_path, 'rb') as f:
        objects = ijson.items(f, '', multiple_values=True)

        for idx, obj in enumerate(objects, 1):
            obsid = obj.get("obsid", None)
            if str(obsid) == str(target_obsid):
                print(f"Found ObsID {target_obsid} at object #{idx}")
                return [obj]
    
    print(f"ObsID {target_obsid} not found.")
    return []

def process_file_robust_with_json_handling(file_path, obsid_target, target, idents, csc_names, tipos, csc_coords, 
                                          all_embeddings, obsids, sources, coords, answers, hardness_ratios, hardness_ratio, bb_kt_list, bb_kt, powlaw_gamma_list, powlaw_gamma, var_index_b_list, var_index_b, var_prob_b_list, var_prob_b, source_flags, source_flag):
    
    try:
        json_objects = stream_search_obsid(file_path, obsid_target)
        
        if not json_objects:
            print("No JSON objects found in file")
            return False
            
        matches_found = 0
        for obj_num, data in enumerate(json_objects, 1):
            try:
                if data.get('obsid') != obsid_target:
                    continue
                matches_found += 1
                
                # target is from the target_obsid_coordinates file, remove if you want to skip this step
                prompt_question1 = f"""The target of the observation is {target}. Is this confirmed by the proposal 
                                    abstract provided, with title {data['proposal']['title']}? If so, please provide
                                    a short context about this observation.
                                    """

                context = data['proposal'].get('abstract', '')
                if not context:
                    print(f"No abstract found for obsid {data['obsid']}")
                    continue
                    
                full_prompt = f"{context}\n\n{prompt_question1}"
                answer = getGPTresponse(full_prompt)

                if answer:
                    print(f" ")
                    print(f"Obsid: {data['obsid']}")
                    print(f"Proposal Title: {data['proposal']['title']}")
                    print(f"Answer: {answer}\n")

                    for j,ido in enumerate(idents):

                            print(f"Source: {ido}")

                            identifiers = Simbad.query_objectids(ido)

                            name_ids = []
                            for idito in identifiers['id'].data:  # was 'ID'
                                if isinstance(idito, bytes):
                                    idito = idito.decode('utf-8')
                                name_ids.append(str(idito))
                            name_ids = list(name_ids)

                            prompt_question2 = f""" 
                            Within the text you are provided with, search for information about the source identified with any of the following names:
                            {', '.join(repr(item) for item in name_ids)}.
                            The source is classified as type {tipos[j]}.
                            
                            Your task is to extract and summarize the physical properties and scientific interpretation of this source as completely and directly as possible, using only information contained in the text.

                            Please first evaluate whether the source is directly mentioned in the text. If the source is mentioned directly (or is the target of the observation), return:  
                            [MENTIONED: YES]  
                            Otherwise return:  
                            [MENTIONED: NO] 
                            Then, provide the full physical summary as before.
                            If the source is not directly mentioned or targeted, provide a general summary based on the information available for sources of type {tipos[j]}.
                            
                            Follow these instructions strictly:
                            ### A) X-ray Properties
                            - Describe variability, including:
                                - Transient behavior, periodicity, flares, quiescence, outbursts
                                - Decay patterns (exponential decay, e-folding times, linear decay rates)
                                - Orbital periods (report estimates if available)
                            - Spectral properties:
                                - Spectral models fitted (e.g., power-law, disk blackbody, Comptonization)
                                - Best-fit parameters (e.g., photon index Γ, disk temperature kT_in, column density N_H)
                                - Include all provided uncertainties with numerical values.
                                - Report state transitions (e.g., hard state, thermally dominated, steep power law)
                                - Hardness ratios (if provided)
                            - Flux measurements and luminosity (always include units where possible)
                            - Timing analysis (variability timescales, periodicities, orbital periods)
                            - Multi-wavelength data (e.g., optical magnitudes, IR, radio measurements if stated)
                            - Include any specific values explicitly reported in the text.
                            
                            ### B) Use in Scientific Hypotheses
                            - Describe how these properties are used to test or constrain scientific models discussed in the text.
                            - Include discussion of accretion processes, black hole or neutron star identification, coronal structure, super-Eddington behavior, binary evolution, or any astrophysical interpretation directly stated.
                            
                            ### Formatting Instructions:
                            - Present your answer in complete sentences, fully written out, but remain clear and concise.
                            - Always prioritize quantitative measurements when available.
                            - Include all physical properties mentioned, even if multiple models or parameters are provided.
                            - Do not speculate beyond the information provided.
                            - Do not refer to the source by name or target name.
                                            """

                            concatenated_context2 = answer
                            max_tokens_allowed = 20000 
                            current_token_count = estimate_token_count(concatenated_context2)

                            papers = data.get('papers', [])
                            print("Number of papers found:", len(papers))
                            
                            for i, paper in enumerate(papers):
                                try:
                                    body_text = paper.get('body', '') + "\n"
                                    if not body_text.strip():
                                        continue
                                        
                                    body_token_count = estimate_token_count(body_text)

                                    if current_token_count + body_token_count > max_tokens_allowed:
                                        print(f"Token limit reached, skipping remaining papers") 
                                        break
                                    else:
                                        concatenated_context2 += body_text
                                        current_token_count += body_token_count

                                except Exception as e:
                                    print(f'Could not concatenate paper {i}: {e}')
                                    continue

                            full_prompt2 = f"{concatenated_context2}\n\n{prompt_question2}"
                            answer2 = getGPTresponse(full_prompt2)

                            # Default to unknown
                            source_flag = 0  
                            # Look for [MENTIONED: YES] tag
                            if re.search(r'\[MENTIONED:\s*YES\]', answer2, re.IGNORECASE):
                                source_flag = 1
                            elif re.search(r'\[MENTIONED:\s*NO\]', answer2, re.IGNORECASE):
                                source_flag = 0
                            else:
                                print("Warning: MENTIONED tag not found — defaulting to 0.")

                            # Add the response to all_answers2 if it is valid
                            if answer2:
                                if (('This source is not discussed' in answer2) or ('this source is not discussed' in answer2)):
                                    continue
                                else:
                                    print(answer2)
                                    # Generate embeddings for the answer
                                    embedding = getEmbeddings(answer2)

                                    if embedding:
                                        all_embeddings.append(embedding)
                                        obsids.append(data['obsid'])
                                        sources.append(csc_names[j])
                                        coords.append(csc_coords[j])
                                        answers.append(answer2)
                                        hardness_ratios.append(hardness_ratio)
                                        bb_kt_list.append(bb_kt)
                                        powlaw_gamma_list.append(powlaw_gamma)
                                        var_index_b_list.append(var_index_b)
                                        var_prob_b_list.append(var_prob_b)
                                        source_flags.append(source_flag)
                            else:
                                print("No answer for obsid: ", data['obsid'])
                
            except Exception as e:
                print(f"Error processing object {obj_num}: {e}")
                continue
                
    except Exception as e:
        print(f"Error opening/processing file {file_path}: {e}")
        return False
    
    return True

def estimate_token_count(text):
    return len(text.split())

def getGPTresponse(prompt):
    response = client.chat.completions.create(
        model=generation_model,
        messages=[
            {"role": "system", "content": "You are an expert astronomer specializing in the Chandra Data Archive."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600  # Adjust based on your needs
    )

    try:
        result = response.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {e}")
        result = None

    return result

def getEmbeddings(text):
    response = client.embeddings.create(
        model=embedding_model,
        input=text
    )

    try:
        embedding = response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        embedding = None

    return embedding

def main(args):
    ini = args.ini_obsid
    end = args.end_obsid

    print('Hello: ', ini, end)

    # List to hold all the embeddings
    all_embeddings = []
    obsids = []
    sources = []
    coords = []
    answers = []
    hardness_ratios = []
    bb_kt_list = []
    powlaw_gamma_list = []
    var_index_b_list = []
    var_prob_b_list = []
    source_flags = []

    for observation in df_bright_obsids['obsid'].values[ini:end]:
        print(f"Processing observation: {observation}")
        
        try:
            # SIMBAD processing section
            obsid_target = observation
            target = df['target'].values[df['obsid'].values == obsid_target][0]
            # Look up correct source name for this obsid
            try:
                source_row = df_bright_obsids[df_bright_obsids['obsid'] == obsid_target]
                source_name = source_row['name'].values[0]
            except:
                print(f"No matching source found for obsid {obsid_target}")
                continue
            print(df['obsid'].values[df['obsid'].values == obsid_target][0])
            print(df['target'].values[df['obsid'].values == obsid_target][0])
        
            # Construct the query that takes each ACIS obsID, and gets coordinates for all the sources in the obsid
            qry = f"""
            SELECT DISTINCT m.name,o.obsid,m.ra,m.dec,o.hard_hs,o.bb_kt,o.powlaw_gamma,o.var_index_b,o.var_prob_b
            FROM csc21.master_source m , csc21.master_stack_assoc a , csc21.observation_source o , 
                 csc21.stack_observation_assoc b , csc21.stack_source s 
            WHERE (m.name = '{source_name}') AND (o.instrument = 'ACIS') 
            AND (o.obsid = {obsid_target}) 
            AND (o.theta <= 10) 
            AND (o.flux_significance_b >= 4) 
            AND (m.name = a.name)
            AND (s.detect_stack_id = a.detect_stack_id and s.region_id = a.region_id) 
            AND (s.detect_stack_id = b.detect_stack_id and s.region_id = b.region_id) 
            AND (o.obsid = b.obsid and o.obi = b.obi and o.region_id = b.region_id)
            ORDER BY name ASC
            """

            # TO QUERY FOR JUST ONE SOURCE: change m.name NOT LIKE... to m.name = '2CXO J######+######'
            
            results = tap.search(qry)
            try:
                hardness_ratio = results[0]['hard_hs']
            except:
                hardness_ratio = None 
            try:
                bb_kt = results[0]['bb_kt']
            except:
                bb_kt = None
            
            try:
                powlaw_gamma = results[0]['powlaw_gamma']
            except:
                powlaw_gamma = None
            
            try:
                var_index_b = results[0]['var_index_b']
            except:
                var_index_b = None
            
            try:
                var_prob_b = results[0]['var_prob_b']
            except:
                var_prob_b = None
            try:
                source_flag = results[0]['source_flag']
            except:
                source_flag = None 

            print(f"Found {len(results)} sources in CSC catalog for obsid {obsid_target}")
        
            # The SIMBAD API is used here to find all the catalog sources from the previous query that have identifiers (e.g., that have been given a name other than the CSC name)
            simbad = Simbad()
            simbad.add_votable_fields("otype")
        
            # Loop through the results and query the region for each
            idents = []
            csc_names = []
            tipos = []
            csc_coords = []
            
            for i in range(len(results)):
                # Perform the query for the given coordinates
                print(f'Processing source {i+1}/{len(results)}')
                print('Coordinates: ', results[i]['ra'], results[i]['dec'])
                print('CSC ID: ', results[i]['name'], '     Obsid: ', results[i]['obsid'])
        
                try:
                    simby = simbad.query_region(
                        SkyCoord(results[i]['ra'], results[i]['dec'], unit=(u.deg, u.deg), frame='fk5'),
                        radius=5 * u.arcsec
                    )
        
                    if simby is not None and hasattr(simby, '__len__') and len(simby) > 0:
                        # Check if required columns exist
                        has_main_id = 'main_id' in simby.colnames
                        has_otype = 'otype' in simby.colnames
                        
                        if has_main_id:
                            main_id = simby['main_id'][0]
                            if isinstance(main_id, bytes):
                                main_id = main_id.decode('utf-8')
                            print('ID: ', main_id)
                            idents.append(main_id)
                        else:
                            print("main_id column not found, skipping this source")
                            continue
                            
                        if has_otype:
                            otype = simby['otype'][0]
                            if isinstance(otype, bytes):
                                otype = otype.decode('utf-8')
                            print('TYPE: ', otype)
                            tipos.append(otype)
                        else:
                            print("otype column not found, using 'Unknown' as type")
                            tipos.append('Unknown')
                            
                        csc_names.append(results[i]['name'])
                        csc_coords.append([results[i]['ra'], results[i]['dec']])
                        
                    else:
                        if simby is None:
                            print("SIMBAD query returned None")
                        else:
                            print(f"SIMBAD query returned empty table (length: {len(simby) if hasattr(simby, '__len__') else 'unknown'})")
                        print(f"No SIMBAD match found for CSC source {results[i]['name']}")
                        
                except Exception as e: # for debugging purposes
                    print(f"Error querying SIMBAD for source {i} ({results[i]['name']}): {e}")
                    print(f"Error type: {type(e).__name__}")
                    continue
                    
                print(' ')
            
        except Exception as e:
            print(f"Error in SIMBAD processing for observation {observation}: {e}")
            print(f"Error type: {type(e).__name__}")
            idents = []
            csc_names = []
            tipos = []
            csc_coords = []
            continue  # Skip this observation and go to the next one
        
        print(f"Starting file processing for ObsID {obsid_target}")
        
        success = process_file_robust_with_json_handling(
            file_path, obsid_target, target, idents, csc_names, tipos, csc_coords,
            all_embeddings, obsids, sources, coords, answers, hardness_ratios, hardness_ratio, bb_kt_list, bb_kt, powlaw_gamma_list, powlaw_gamma, var_index_b_list, var_index_b, var_prob_b_list, var_prob_b, source_flags, source_flag
        )
        
        if not success:
            print(f"Failed to process file for obsid {obsid_target}")
            continue

    # Convert the list of embeddings to a numpy array
    embeddings_array = np.array(all_embeddings)

    print('Shape of embeddings: ',np.shape(embeddings_array))

    ## The embeddings are saved in this file
    np.save(f'prompt_7_all_embeddings_simbad_test5_{ini}_{end}.npy',all_embeddings)
    metadata_entries = []
    for i in range(len(obsids)):
        entry = {
            'obsid': obsids[i],
            'source': sources[i],
            'ra': coords[i][0],
            'dec': coords[i][1],
            'otype': tipos[i] if i < len(tipos) else 'Unknown',
            'hardness_ratio': hardness_ratios[i],
            'bb_kt': bb_kt_list[i],
            'powlaw_gamma': powlaw_gamma_list[i],
            'var_index_b': var_index_b_list[i],
            'var_prob_b': var_prob_b_list[i],
            'source_flag': source_flags[i],
            'answer': answers[i]
        }
        metadata_entries.append(entry)
    
    # Convert to DataFrame and save
    metadata_df = pd.DataFrame(metadata_entries)
    metadata_df.to_csv(f'prompt_7_embeddings_metadata_{ini}_{end}.csv', index=False)
    print(f'Metadata saved to embeddings_metadata_{ini}_{end}.csv')

    ## Check that all lists are of the same length
    assert len(all_embeddings) == len(answers), \
        "All input lists must have the same length."

    # Build the 'data' structure
    data = []
    print('Obsids: ',obsids)
    print('Sources: ',sources)

    for i in range(len(obsids)):
        entry = {
            'obsid': obsids[i],
            'src': sources[i],
            'coords': coords[i],
            'embedding': all_embeddings[i],
            'answer': answers[i]
        }
        data.append(entry)

    # Save data to an HDF5 file
    filename = f'prompt_7_text_embeddings_simbad_test5_{ini}_{end}.h5'
    print(filename)

    with h5py.File(filename, 'w') as f:
        for i, entry in enumerate(data):
            group = f.create_group(f'entry_{i}')
            group.create_dataset('obsid', data=entry['obsid'])
            group.create_dataset('source', data=entry['src'])
            group.create_dataset('coords', data=entry['coords'])
            group.create_dataset('embedding', data=entry['embedding'])
            group.create_dataset('answer', data=entry['answer'].encode('utf-8'))

    print(f"Data saved to {filename}.")

    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs GPT text embeddings')
    parser.add_argument('--ini_obsid', type=int, default=0, help='Index of initial obsid')
    parser.add_argument('--end_obsid', type=int, default=100, help='Index of final obsid')
    #args = parser.parse_args() for submitting jobs
    # for in-line output:
    args = parser.parse_args(args=['--ini_obsid', '0', '--end_obsid', '10'])

    main(args)
