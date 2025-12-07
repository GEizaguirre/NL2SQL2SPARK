import dotenv

from spark_nl import (
    get_spark_session,
    get_spark_sql,
    get_spark_agent,
    run_nl_query,
    process_result,
    print_results,
    pretty_print_cot,
    run_sparksql_query
)
from benchmark_ds import (
    load_tables,
    load_query_info
)
from llm import get_llm
from evaluation import (
    translate_sqlite_to_spark,
    jaccard_index,
    result_to_obj,
    evaluate_spark_sql
)

QUERY_ID = 2

def benchmark_query():

    dotenv.load_dotenv()

    spark_session = get_spark_session()

    database_name, nl_query, golden_query = load_query_info(QUERY_ID)
    golden_query_spark = translate_sqlite_to_spark(golden_query)
    print(f"--- Benchmarking Query ID {QUERY_ID} on Database '{database_name}' ---")
    print(f"NL Query: {nl_query}")
    load_tables(spark_session, database_name)
    spark_sql = get_spark_sql()
    llm = get_llm()
    agent = get_spark_agent(spark_sql, llm=llm)
    run_nl_query(agent, nl_query)
    json_result = process_result()
    print_results(json_result)
    # pretty_print_cot(json_result)

    if json_result["execution_status"] == "VALID":
        ground_truth_df = run_sparksql_query(spark_session, golden_query_spark)
        print(f"Golden Query (Spark SQL): \033[93m{golden_query_spark}\033[0m")
        print("Ground Truth:")
        ground_truth_df.show()

        # Execution Accuracy
        inferred_result = json_result["query_result"]
        ea = jaccard_index(ground_truth_df, inferred_result)
        print(f"Jaccard Index: {ea}")

        # Structural Accuracy        
        spark_sql_query = json_result.get("sparksql_query")
        if spark_sql_query:
            em_score = evaluate_spark_sql(golden_query_spark, spark_sql_query, spark_session)
            print(f"Spider Exact Match Score: {em_score}")

if __name__ == "__main__":
    benchmark_query()
