-- Drop dataload table and create new one if exists
drop table if exists dataload;
create table dataload(layer_id integer, poly nvarchar(10000), label nvarchar(100));
LOAD DATA INFILE 'gds_polygon_and_label_try.txt' INTO TABLE dataload fields terminated by ';'  ENCLOSED BY '' LINES TERMINATED BY '\n';

-- drop input table and create new table to extract polygon information from dataload table
drop table if exists gds_input;
create table gds_input(id integer, layer_id integer, poly geometry, label varchar(100));
insert into gds_input(select distinct row_number() over() id, a.layer_id, st_geomfromtext(a.poly) poly, a.label  from(select distinct layer_id, poly, label from dataload)a);

-- create gds relationship table
drop table if exists gds_relationship;
create table gds_relationship(source_id integer, source_label nvarchar(100),dest_id integer, dest_label nvarchar(100) );-- creates table for above layer and below layer to find 

drop table if exists via_layers;
create table via_layers(via_layer integer, above_layer integer, below_layer integer);
insert into via_layers values(50, 49, 51);
insert into via_layers values(61, 51, 62);

delimiter //
drop procedure if exists create_relationships()//
create procedure create_relationships()
begin
	declare num_via_layers int;
	declare i int default 0;
	select count(*) into num_via_layers from via_layers;
	while i < num_via_layers do 
		drop table if exists above_layer;
		drop table if exists below_layer;
		create temporary table above_layer(source_id integer, source_label nvarchar(1000), source_poly geometry, 
		dest_id integer,dest_label nvarchar(1000), dest_poly geometry);
		create temporary table below_layer(source_id integer, source_label nvarchar(1000), source_poly geometry, 
		dest_id integer,dest_label nvarchar(1000), dest_poly geometry);
		
		-- computing the overlap between the above layer of the via layer and via layer itself
		insert into above_layer (
		select a.id source_id, a.label source_label, a.poly source_poly, b.id dest_id, b.label dest_label, b.poly dest_poly from
		(select id,label,poly,layer_id from gds_input where layer_id in (select above_layer from via_layers order by via_layer limit i, 1))a,
		(select id,label,poly,layer_id from gds_input where layer_id in (select via_layer from via_layers order by via_layer limit i, 1))b
		where a.id<>b.id and st_intersects(a.poly, b.poly));
		-- computing the overlap between the below layer of the via layer and via layer itself
		insert into below_layer (
		select a.id source_id, a.label source_label, a.poly source_poly, b.id dest_id, b.label dest_label, b.poly dest_poly from
		(select id,label,poly,layer_id from gds_input where layer_id in (select below_layer from via_layers order by via_layer limit i, 1))a,
		(select id,label,poly,layer_id from gds_input where layer_id in (select via_layer from via_layers order by via_layer limit i, 1))b
		where a.id<>b.id and st_intersects(a.poly, b.poly));

		-- unidirectional edge
		insert into gds_relationship(
		select distinct a.source_id source_id,a.source_label source_label, b.source_id dest_id,b.source_label dest_label from
		(select * from above_layer)a,
		(select * from below_layer)b
		where a.dest_id=b.dest_id);

		-- creating the same edge in other direction
		insert into gds_relationship(
		select distinct b.source_id source_id,b.source_label source_label, a.source_id dest_id,a.source_label dest_label from
		(select * from above_layer)a,
		(select * from below_layer)b
		where a.dest_id=b.dest_id);
		set i = i+1;
	end while;
end //
delimiter;

call create_relationships();

create or replace view reordered_node as(
select a.node,a.label,row_number() over()-1 as id from
(select distinct a.node node, a.label label from 
(select distinct source_id node,source_label label from gds_relationship 
union
select distinct dest_id node,dest_label label from gds_relationship)a
order by a.node,a.label)a);


-- persists relationships into a file with reordered ids starting from 0
-- the IDs are reordered as the algorithm expects the ids to be starting from 0
select distinct a.source_id,b.id dest_id from
(select distinct b.id source_id, a.dest_id from
(select distinct source_id,dest_id from gds_relationship)a,
(select node,id from reordered_node)b
where a.source_id=b.node)a,
(select node, id from reordered_node)b
where a.dest_id=b.node
INTO OUTFILE 'relationship_final_mysql_nodes_reordered.txt' 
FIELDS ENCLOSED BY '' 
TERMINATED BY ' ' 
LINES TERMINATED BY '\n';

-- persists labels along with id of each which are reordered starting from 0
select distinct id, label from reordered_node
INTO OUTFILE 'node_labels_final_reordered.txt' 
FIELDS ENCLOSED BY '' 
TERMINATED BY ' ' 
LINES TERMINATED BY '\n';

-- map of the id for each node in the graph
select distinct node,id from reordered_node
INTO OUTFILE 'node_id_map_reordered.txt' 
FIELDS ENCLOSED BY '' 
TERMINATED BY ' ' 
LINES TERMINATED BY '\n';


-- persists the node with features width, length, orientation of the cell as features
select distinct a.id,a.width,a.length,a.x_orientation,a.y_orientation from (select distinct b.id id,
round(a.width, 2) width,
round(a.length, 2) length, 
case when a.x_orientation is null then 90 else round(a.x_orientation,2) end x_orientation,
case when a.y_orientation is null then 90 else round(a.y_orientation,2)  end y_orientation from
(select abs(st_y(st_pointn(st_exteriorring(poly), 1))- st_y(st_pointn(st_exteriorring(poly),2))) width,
abs(st_x(st_pointn(st_exteriorring(poly), 2))- st_x(st_pointn(st_exteriorring(poly),3))) length,
id,
(tan(st_x(st_pointn(st_exteriorring(poly), 2))/ st_x(st_pointn(st_exteriorring(poly),3)))*180)/pi() as x_orientation,
(tan(st_y(st_pointn(st_exteriorring(poly), 1))/ st_y(st_pointn(st_exteriorring(poly),2)))*180)/pi() as y_orientation,
layer_id from gds_input)a,	
(select node, id from reordered_node)b
where a.id=b.node)a
INTO OUTFILE 'node_features_length_reordered.txt' 
FIELDS ENCLOSED BY '' 
TERMINATED BY ' ' 
LINES TERMINATED BY '\n';
