File descriptions:

1) dose-form-stoplist.tsv -- all dose-form mentions in FAERS data from 2012 - 2021

Query:
select distinct upper(dose_form) dose_form
from faers.drug
order by dose_form
;



2) dose-unit-stoplist.tsv -- all dose unit mentions in FAERS data from 2012 - 2021 

Query (with removal of spaces at the beginning of lines and single quotes on the results):
select distinct upper(dose_unit) dose_unit
from faers.drug
order by dose_unit
;


3) positive-unmapped-pairs-gsrs-name-to-latin-binomial.tsv -- POSITIVE unmapped pairs G-SRS to Latin binomial

Query: The setup up queries (below) and

select name, latin_binomial
from scratch_embedding.unmapped_nps
;

4) positive-unmapped-pairs-gsrs-name-to-common-name.tsv -- POSITIVE unmapped pairs G-SRS to Latin binomial common name

Query: The setup up queries (below) and

select name, common_name
from scratch_embedding.unmapped_nps
;


5) positive-unmapped-pairs-common-name-or-latin-binomial-copies.tsv -- POSITIVE unmapped pairs common-name to common name, common name to Latin binomial and vice versa, Latin binomial to Latin binomial


select distinct *
from (
 select common_name, latin_binomial
 from scratch_embedding.unmapped_nps
 where common_name != ''
 union 
 select latin_binomial, common_name
 from scratch_embedding.unmapped_nps
 where common_name != ''
 union 
 select common_name, common_name
 from scratch_embedding.unmapped_nps
 where common_name != ''
 union 
 select latin_binomial, latin_binomial
 from scratch_embedding.unmapped_nps
) t
;

6) negative-unmapped-pairs-all.tsv -- NEGATIVE pairs for the umpapped NP strings from GSRS

with pos_1 as ( 
 select name p1, common_name p2
 from scratch_embedding.unmapped_nps
), pos_2 as ( 
 select name p1, latin_binomial p2
 from scratch_embedding.unmapped_nps
), pos_3 as (
 select distinct common_name p1, latin_binomial p2
 from (
  select common_name, latin_binomial
  from scratch_embedding.unmapped_nps
  where common_name != ''
  union 
  select latin_binomial, common_name
  from scratch_embedding.unmapped_nps
  where common_name != ''
  union 
  select common_name, common_name
  from scratch_embedding.unmapped_nps
  where common_name != ''
  union 
  select latin_binomial, latin_binomial
  from scratch_embedding.unmapped_nps
 ) t
), pos_4 as ( 
 select distinct upper(regexp_replace(c1.concept_name, '\[.*',''))  p1, c1.concept_class_id p2
 from staging_vocabulary.concept c1 
  inner join scratch_sanya.np_faers_reference_set rs on c1.concept_class_id = rs.related_common_name
), pos_all as (
 select p1,p2 from pos_1
 union 
 select p1,p2 from pos_2
 union 
 select p1,p2 from pos_3
 union 
 select p1,p2 from pos_4
), rand_1 as ( 
 select p1, row_number() over (order by random()) row_id from pos_all
), rand_2 as ( 
 select p2, row_number() over (order by random()) row_id from pos_all
)
select distinct upper(rand_1.p1) p1, upper(rand_2.p2)
from rand_1 inner join rand_2 on rand_1.row_id = rand_2.row_id
where p1 != '' and p1 is not null
 and p2 != '' and p2 is not null
except 
select upper(p1) p1, upper(p2) p2
from pos_all
;


7) NP_FAERS_mapped_20220215.csv -- the manually create references set for ~70 drugs

8) upper_unmap_orig-_drug_names_no_model_overlap_20220224.csv -- the drugname strings that we need to run through the final model for novel discovery of NP mappings -- this comes from the combined_drug_mapping table where the lookup_value is null and then subtracting rows that contain strings we have mapped in the reference set

9) NP_FAERS_negative_pairs_20220222.csv -- the negative pairs created by random sampling from the NP_FAERS_mapped_20220215.csv

10) translation_test_nps_202203171038.csv -- ~6000 NP strings that can be used for translation testing but highly NP focused and not for novelty





-------------------

NOTE: "set up queries" for (3), (4), (5), and (6)

set search_path to scratch_feb2022_np_vocab;
SELECT dblink_connect('g_substance_reg', 'dbname=g_substance_reg user=''rw_grp'' password=''rw_grp''');  -- NOTE: edit username and pword

with remote_lb as (
   SELECT * FROM dblink('g_substance_reg', 
   						'select related_latin_binomial, related_common_name, substance_uuid, name, type, organism_genus, organism_species 
						 from scratch_sanya.test_srs_np') 
   						AS test_srs_np_reduced_cols(latin_binomial varchar, common_name varchar, substance_uuid varchar, name varchar, 
													type varchar, organism_genus varchar, organism_species varchar)
)
select *
into scratch_feb2022_np_vocab.test_srs_np_reduced_cols
from remote_lb
;
-- 12381

select *
from 
scratch_feb2022_np_vocab.test_srs_np_reduced_cols
;

select distinct upper(common_name), name
from scratch_feb2022_np_vocab.test_srs_np_reduced_cols
where common_name is not null and common_name != ''
;


-- unmapped common names 
with t1 as (
  select distinct upper(regexp_replace(c1.concept_name, '\[.*','')) np_string, c1.concept_class_id 
  from staging_vocabulary.concept c1 
   inner join scratch_sanya.np_faers_reference_set rs on c1.concept_class_id = rs.related_common_name
  order by concept_class_id
), gsrs_common_names as ( 
  select distinct upper(common_name) common_name, name, concat(organism_genus, ' ', organism_species) latin_binomial
  from  scratch_feb2022_np_vocab.test_srs_np_reduced_cols
  where common_name is not null and common_name != ''
     and organism_genus is not null and organism_species is not null
)
select *
-- into scratch_embedding.unmapped_nps
from gsrs_common_names 
where common_name not in (select np_string from t1)
  and name not in (select np_string from t1)
order by common_name
;
-- 199



with t1 as (
  select distinct upper(regexp_replace(c1.concept_name, '\[.*','')) np_string, c1.concept_class_id 
  from staging_vocabulary.concept c1 
   inner join scratch_sanya.np_faers_reference_set rs on c1.concept_class_id = rs.related_common_name
  order by concept_class_id
), gsrs_names_w_no_common as ( 
  select distinct upper(common_name) common_name, name, concat(organism_genus, ' ', organism_species) latin_binomial
  from  scratch_feb2022_np_vocab.test_srs_np_reduced_cols
  where common_name is null or common_name = ''
     and organism_genus is not null and organism_species is not null
)
-- select count(*)
-- insert into scratch_embedding.unmapped_nps
select *
from gsrs_names_w_no_common
where name not in (select np_string from t1)
order by latin_binomial
;
-- 10639

-- 10838
select count(*) from scratch_embedding.unmapped_nps;
