drop table if exists relationship;
-- table definition
create table if not exists relationship(
	experiment_id	integer not null default 0,
    region_id		integer	not null default 0,
	source_mfield	integer not null,
	source_sfield	integer not null,
	dest_mfield		integer not null,
	dest_sfield		integer not null,
	x_centroid		double not null,
	y_centroid		double not null,
	contrast        integer not null default -1,
	image_path		nvarchar(1000) not null,
	unique key		`id`		(experiment_id, region_id, source_mfield, source_sfield, dest_mfield, dest_sfield),
	index			`relations`	(experiment_id, region_id, source_mfield, source_sfield),
	index			`loc_relations` (experiment_id, region_id, source_mfield, x_centroid, y_centroid),
	index           `contrast_relation` (experiment_id, region_id, source_mfield, source_sfield, contrast)
);

-- data load into table relationship
LOAD DATA INFILE 'relationship.csv' INTO TABLE relationship fields terminated by ';' ENCLOSED BY '"'
LINES TERMINATED BY '\n';
