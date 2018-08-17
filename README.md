all mySQL files are setup in 
C:\ProgramData\MySQL\MySQL Server 8.0\Data\msem_db

from msql console, use execute the following command:

D:\projects\graph_analysis\DataloadScript.sql

You can modify DataloadScript to add more columns such as contrast, brightness, etc. 
You also need to add index to these new added parameters.

// graph based analysis tools
use klayout to open and view write_buffer.gds, which contains the circuit layout info
 
> python read_from_gds.py  // internally has the path to write_buffer.gds
this decode info from gds file and  write to a file:
data/gds_polygon_and_label_try.txt
you need to copy the above file to C:\ProgramData\MySQL\MySQL Server 8.0\Data\msem_db

from mysql console, run 
extract_relationships.sql
which will load in data/gds_polygon_and_label_try.txt and compute all spatial overlap between polygons
outputs to C:\ProgramData\MySQL\MySQL Server 8.0\Data\msem_db are:
 
node_features_length_reordered.txt
node_labels_final_reordered.txt
relationship_final_mysql_nodes_reordered.txt

all GCN model and data are saved in the folder:
graph_analysis/GCN

ind.gds.tx and ind.gds.ty are the test data for GCN
ind.gds.x and ind.gds.y are the training data for GCN
to generate these, run the following command to install the GCN package
> python setup.py 
To invoke the data processing pipeline for the algorithm
> python dataparse_gcn.py

to train and test, 
run GCN/train.py
The model is saved in models.python

2nd approach is to use deepwalk, which takes relationship_final_mysql_nodes_reordered.txt

python deepwalk_fullyconnected.py

graph_clustering_gds is a clustering method using autoencoder-decoder approach
input is the similarity matrix output
