#include <stdlib.h>
#include <iostream>

#include <chrono>

#include "test.h"
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>
#define HOST "localhost"
#define USER "root"
#define PASS "admin"
#define DB "test_index"
using namespace std;

int mysqltest(int argc, char *argv[])
{
	int experiment_id, source_mfield, region_id;
	char querySField;
	cout << "Do you want to query on sfield? Y|n" << endl;
	cin >> querySField;

	cout << "Enter the experiment id" << endl;
	cin >> experiment_id;
	
	cout << "Enter the region id" << endl;
	cin >> region_id;

	cout << "Enter the source mfield" << endl;
	cin >> source_mfield;
	
	int source_sfield;
	bool isquerySField = false;
	if (querySField == 'Y' || querySField == 'y') {
		cout << "Enter the source sfield" << endl;
		cin >> source_sfield;
		isquerySField = true;
	}
	const string url = HOST;
	const string user = USER;
	const string pass = PASS;
	const string database = DB;
	sql::Driver* driver = get_driver_instance();
	std::auto_ptr<sql::Connection> con(driver->connect(url, user, pass));
	
	sql::Statement *stmt = NULL;
	sql::ResultSet  *res = NULL;
	sql::PreparedStatement *prep_stmt = NULL;
	con->setSchema(database);
	if (!isquerySField) {
		auto start = std::chrono::system_clock::now();
		prep_stmt = con->prepareStatement("SELECT source_sfield, dest_mfield, dest_sfield FROM relationship where experiment_id = ? and region_id =? and source_mfield=?");
		prep_stmt->setInt(1, experiment_id);
		prep_stmt->setInt(2, region_id);
		prep_stmt->setInt(3, source_mfield);
		res = prep_stmt->executeQuery();
		while (res->next()) {
			// You can use either numeric offsets...
		//	cout << "source_sfield = " <<
			res->getInt(1); // getInt(1) returns the first column
														  // ... or column names for accessing results.
		//	cout << ", dest_mfield = " << 
			res->getString("dest_mfield");
		//	cout << ", dest_sfield = " << 
			res->getString("dest_sfield");// << endl;
		} 
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - start;
		std::cout << "Time to execute query is " << " ints : " << diff.count() << " s\n";
	}
	else { 
		auto start = std::chrono::system_clock::now();
		prep_stmt = con->prepareStatement("SELECT source_sfield, dest_mfield, dest_sfield FROM relationship where experiment_id = ? and region_id =? and source_mfield=? and source_sfield = ?");
		prep_stmt->setInt(1, experiment_id);
		prep_stmt->setInt(2, region_id);
		prep_stmt->setInt(3, source_mfield);
		prep_stmt->setInt(4, source_sfield);
		res = prep_stmt->executeQuery();
		while (res->next()) {
			// You can use either numeric offsets...
			cout << "source_sfield = " << res->getInt(1); // getInt(1) returns the first column
														  // ... or column names for accessing results.
			cout << ", dest_mfield = " << res->getString("dest_mfield");
			cout << ", dest_sfield = " << res->getString("dest_sfield") << endl;
		}
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - start;
		std::cout << "Time to execute query is " << " ints : " << diff.count() << " s\n";
	}
	
	//system("pause");
	printf("test close 1\n");
	delete res;
	printf("test close 2\n");
	delete stmt;
	printf("test close 3\n");
	con->close();
	printf("test close 4\n");
	return 0;
}