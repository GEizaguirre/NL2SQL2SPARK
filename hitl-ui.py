import os
import sys
import dotenv
import argparse

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from spark_nl import (
    get_spark_session,
    get_spark_sql,
    get_spark_agent,
    run_nl_query,
    process_result,
    print_results,
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
    evaluate_spark_sql
)

def benchmark_nl_query(query_id, user_nl_query, provider):
    dotenv.load_dotenv()

    spark_session = get_spark_session()

    # Load context (DB name, golden query) from the benchmark dataset
    database_name, original_nl_query, golden_query = load_query_info(query_id)
    golden_query_spark = translate_sqlite_to_spark(golden_query)
    
    print(f"--- Benchmarking Query ID {query_id} on Database '{database_name}' ---")
    print(f"Original NL Query (from DB): {original_nl_query}")
    print(f"User NL Query (from CLI): {user_nl_query}")
    
    load_tables(spark_session, database_name)
    spark_sql = get_spark_sql()
    llm = get_llm(provider=provider)
    agent = get_spark_agent(spark_sql, llm=llm)
    
    # Run the agent with the USER provided NL query
    run_nl_query(agent, user_nl_query)
    json_result = process_result()
    print_results(json_result)

    print(f"User NL Query (from CLI): {user_nl_query}")
    print(f"Golden Query (Spark SQL): \033[93m{golden_query_spark}\033[0m")

    # Evaluation
    if json_result["execution_status"] == "VALID":
        ground_truth_df = run_sparksql_query(spark_session, golden_query_spark)
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
    parser = argparse.ArgumentParser(description="Run NL2SQL workflow with a custom NL query.")
    parser.add_argument("--id", type=int, default=2, help="Query ID from the benchmark (for context/ground truth)")
    parser.add_argument("--nl-query", type=str, required=True, help="The Natural Language Query to process")
    parser.add_argument("--provider", type=str, default="google", help="LLM provider (default: google)")
    
    args = parser.parse_args()
    
    benchmark_nl_query(args.id, args.nl_query, args.provider)
