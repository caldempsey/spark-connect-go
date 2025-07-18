//
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"context"
	"flag"
	"fmt"
	"log"

	"github.com/apache/spark-connect-go/v40/spark/sql/types"

	"github.com/apache/spark-connect-go/v40/spark/sql/functions"

	"github.com/apache/spark-connect-go/v40/spark/sql"
	"github.com/apache/spark-connect-go/v40/spark/sql/utils"
)

var (
	remote = flag.String("remote", "sc://localhost:15002",
		"the remote address of Spark Connect server to connect to")

	filedir = flag.String("filedir", "/tmp",
		"the root directory to save the files generated by this example program")
)

func main() {
	flag.Parse()
	ctx := context.Background()
	spark, err := sql.NewSessionBuilder().Remote(*remote).Build(ctx)
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}
	defer utils.WarnOnError(spark.Stop, func(err error) {})

	df, err := spark.Sql(ctx, "select id from range(100)")
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	df, _ = df.FilterByString(ctx, "id < 10")
	err = df.Show(ctx, 100, false)
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	df, err = spark.Sql(ctx, "select * from range(100)")
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	df, _ = df.Filter(ctx, functions.Col("id").Lt(functions.Expr("10")))
	err = df.Show(ctx, 100, false)
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	df, _ = spark.Sql(ctx, "select * from range(100)")
	df, err = df.Filter(ctx, functions.Col("id").Lt(functions.Lit(types.Int64(20))))
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}
	err = df.Show(ctx, 100, false)
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	df, err = spark.Sql(ctx, "select 'apple' as word, 123 as count union all select 'orange' as word, 456 as count")
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	log.Printf("DataFrame from sql: select 'apple' as word, 123 as count union all select 'orange' as word, 456 as count")
	err = df.Show(ctx, 100, false)
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	schema, err := df.Schema(ctx)
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	for _, f := range schema.Fields {
		log.Printf("Field in dataframe schema: %s - %s", f.Name, f.DataType.TypeName())
	}

	rows, err := df.Collect(ctx)
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	schema, err = df.Schema(ctx)
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	for _, f := range schema.Fields {
		log.Printf("Field in row: %s - %s", f.Name, f.DataType.TypeName())
	}

	for _, row := range rows {
		log.Printf("Row: %v", row)
	}

	err = df.Writer().Mode("overwrite").
		Format("parquet").
		Save(ctx, fmt.Sprintf("file://%s/spark-connect-write-example-output.parquet", *filedir))
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	df, err = spark.Read().Format("parquet").
		Load(fmt.Sprintf("file://%s/spark-connect-write-example-output.parquet", *filedir))
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	log.Printf("DataFrame from reading parquet")
	err = df.Show(ctx, 100, false)
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	err = df.CreateTempView(ctx, "view1", true, false)
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	df, err = spark.Sql(ctx, "select count, word from view1 order by count")
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	log.Printf("DataFrame from sql: select count, word from view1 order by count")
	err = df.Show(ctx, 100, false)
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	log.Printf("Repartition with one partition")
	df, err = df.Repartition(ctx, 1, nil)
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	err = df.Writer().Mode("overwrite").
		Format("parquet").
		Save(ctx, fmt.Sprintf("file://%s/spark-connect-write-example-output-one-partition.parquet", *filedir))
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	log.Printf("Repartition with two partitions")
	df, err = df.Repartition(ctx, 2, nil)
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	err = df.Writer().Mode("overwrite").
		Format("parquet").
		Save(ctx, fmt.Sprintf("file://%s/spark-connect-write-example-output-two-partition.parquet", *filedir))
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	log.Printf("Repartition with columns")
	df, err = df.Repartition(ctx, 0, []string{"word", "count"})
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	err = df.Writer().Mode("overwrite").
		Format("parquet").
		Save(ctx, fmt.Sprintf("file://%s/spark-connect-write-example-output-repartition-with-column.parquet", *filedir))
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	log.Printf("Repartition by range with columns")
	df, err = df.RepartitionByRange(ctx, 0, functions.Col("word").Desc())
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}

	err = df.Writer().Mode("overwrite").
		Format("parquet").
		Save(ctx, fmt.Sprintf("file:///%s/spark-connect-write-example-output-repartition-by-range-with-column.parquet", *filedir))
	if err != nil {
		log.Fatalf("Failed: %s", err)
	}
}
