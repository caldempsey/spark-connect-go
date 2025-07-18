// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package integration

import (
	"context"
	"testing"

	"github.com/apache/spark-connect-go/v40/spark/sql/types"

	"github.com/apache/spark-connect-go/v40/spark/sql/functions"

	"github.com/apache/spark-connect-go/v40/spark/sql"
	"github.com/stretchr/testify/assert"
)

func TestIntegration_BuiltinFunctions(t *testing.T) {
	ctx := context.Background()
	spark, err := sql.NewSessionBuilder().Remote("sc://localhost").Build(ctx)
	if err != nil {
		t.Fatal(err)
	}

	df, _ := spark.Sql(ctx, "select '[2]' as a from range(10)")
	df, _ = df.Filter(ctx, functions.JsonArrayLength(functions.Col("a")).Eq(functions.IntLit(1)))
	res, err := df.Collect(ctx)
	assert.NoError(t, err)
	assert.Equal(t, 10, len(res))
}

func TestAggregationFunctions_Agg(t *testing.T) {
	ctx, spark := connect()
	df, err := spark.Sql(ctx, "select id, 1, 2, 3 from range(100)")
	assert.NoError(t, err)

	res, err := df.Agg(ctx, functions.Count(functions.Col("id")))
	assert.NoError(t, err)
	cnt, err := res.Count(ctx)
	assert.NoError(t, err)
	assert.Equal(t, int64(1), cnt)

	res, err = df.AggWithMap(ctx, map[string]string{"id": "sum"})
	assert.NoError(t, err)
	rows, err := res.Collect(ctx)
	assert.NoError(t, err)
	assert.Len(t, rows, 1)
	assert.Equal(t, int64(4950), rows[0].At(0))
}

func TestIntegration_ColumnGetItem(t *testing.T) {
	ctx := context.Background()
	spark, err := sql.NewSessionBuilder().Remote("sc://localhost").Build(ctx)
	if err != nil {
		t.Fatal(err)
	}

	df, _ := spark.Sql(ctx, "select sequence(1,10) as s")
	df, err = df.Select(ctx, functions.Col("s").GetItem(types.Int64(2)))
	assert.NoError(t, err)
	res, err := df.Collect(ctx)
	assert.NoError(t, err)
	assert.Equal(t, int32(3), res[0].Values()[0])
}
