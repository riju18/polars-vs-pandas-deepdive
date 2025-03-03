# %% [markdown]
# # Task to experiment
# 1. Fetch data with schema overwriting
# 2. Do some basic analysis
# 3. Best time to use `polars`
# 4. Compare with `pandas` in respect of both time and efficiency
# 
# ----------------------------------
# - *Calculate time for each task*
# ----------------------------------

# %%
import time
import polars as pl
import pandas as pd
import numpy as np

# %% [markdown]
# # 1. Read data as dataframe

# %% [markdown]
# - Get csv data (~103 MB)
# - Overwrite schema to make sure proper datatype
# - `ignore` infer_schema: it scans all rows to find the proper datatype. That's why it slow and risky for large dataset 

# %% [markdown]
# - polars code

# %%
start_time = time.time()

# %%
df = pl.read_csv(source='data8277.csv'
                 , has_header=True
                 , separator=','
                 , try_parse_dates=True
                 , schema_overrides={"count": pl.Int32}
                #  , infer_schema=True  # costly: traberse all rows to find out correct data type
                 , ignore_errors=True
                 , encoding='utf8')

# %%
execution_time = time.time() - start_time
print(f"Time to fetch the csv file: {execution_time}")

# %%
df.head()

# %% [markdown]
# - pandas code

# %%
start_time_pd = time.time()

# %%
df_pd = pd.read_csv(filepath_or_buffer='data8277.csv', 
                header=0, 
                delimiter=',', 
                parse_dates=True,  
                encoding='utf8')

# %%
execution_time_pd = time.time() - start_time_pd
print(f"Time to fetch the csv file: {execution_time_pd}")

# %%
df_pd.head(5)

# %% [markdown]
# - findings
#     - super fast to parse data 
#     - supports polars native data type, not external numpy based datatype
#     - try `parse dates param` makes it very efficient to detect datetime related col
#     - ignore errors param helps to prevent to break the code while retrieving data

# %%
df.glimpse()  # a snapshot of data

# %% [markdown]
# # 2. Basic data analysis

# %% [markdown]
# a. get specific columns
# 
# - select only cols
# - basic calculations with selected cols
# 
# b. create dereived col
# 
# - to make a derive col from str input as condition use `lit`
# 
# c. filter
# 
# - basic filtereing
# - range
# 
# d. sort
# 
# e. group by
# 
# f. combining DF

# %% [markdown]
# ## 2.a: Selecting cols

# %% [markdown]
# - polars code

# %%
start_time_selecting_cols = time.time()

# %%
df_year_age = df.select(['Year', 'Age'])
df_year_age.head()

# %%
df_year_age = df.select(
    pl.col('Year')
    , (pl.col('Age') * 1.0).alias('Age*1.0')
)
df_year_age.head()

# %%
execution_time_for_selecting_cols = time.time() - start_time_selecting_cols
print(f"execution time for selecting cols: {execution_time_for_selecting_cols}")

# %% [markdown]
# - pandas code: `wip...`

# %% [markdown]
# ## 2.b: Derive col

# %% [markdown]
# - polars code

# %%
start_time_creating_derive_cols = time.time()

# %%
df_derive = df.with_columns(
    gender = pl.when(pl.col("Sex") == 1)
    .then(pl.lit('male'))
    .when(pl.col("Sex") == 2)
    .then(pl.lit('female'))
    .otherwise(pl.lit('others'))
    )

df_derive = df_derive.drop(['Sex'])
df_derive.head()

# %%
execution_time_creating_derive_cols = time.time() - start_time_creating_derive_cols
print(f"time to create derive col: {execution_time_creating_derive_cols}")

# %% [markdown]
# - pandas code: `wip...`

# %% [markdown]
# ## 2.c: Filter

# %% [markdown]
# - polars code

# %%
start_time_for_filtering = time.time()

# %%
df_basic_filter = df_derive.filter(
        df_derive['Year'] < 2007
    )

df_basic_filter.head()

# %%
df_basic_filter_range = df_derive.filter(
    df_derive['Year'].is_between(2006, 2013)  # upper limit inclusive
    )

df_basic_filter_range.head()

# %%
execution_time_for_filtering = time.time() - start_time_for_filtering
print(f"time to filter data: {execution_time_for_filtering}")

# %% [markdown]
# - pandas code: `wip...`

# %% [markdown]
# ## 2.d: Sort

# %% [markdown]
# - polars code

# %%
start_time_for_sorting = time.time()

# %%
df_derive = df_derive.sort(by=["Year", "count"],
                            nulls_last=True)
df_derive.head()

# %%
execution_time_for_sorting = time.time() - start_time_for_sorting
print(f"Time to sort data: {execution_time_for_sorting}")

# %% [markdown]
# - pandas code: `wip...`

# %% [markdown]
# ## 2.e: Group by

# %%
start_time_for_grouping = time.time()

# %%
df_year_wise_count = df_derive.group_by(
    ["Year"],
    maintain_order=True
    ).agg(
        pl.col("count"). \
        sum(). \
        alias('year_wise_total_count'),

        pl.col("count"). \
        mean(). \
        round(2). \
        alias('year_wise_avg_count'),

        pl.col('gender')
        )

df_year_wise_count.head()


# %%
execution_time_for_grouping = time.time() - start_time_for_grouping
print(f"time to aggregate data: {execution_time_for_grouping}")

# %% [markdown]
# - pandas code: `wip...`

# %% [markdown]
# ## 2.f: combining DF
# 
# i. joining. [doc](https://docs.pola.rs/user-guide/transformations/joins/#quick-reference-table)
# 
# ii. concat

# %%
df2 = pl.DataFrame(
    {
        "Year": [2006, 2013, 2018, 2019],
    }
)

df2.head()

# %% [markdown]
# ## 2.f.i: joining

# %% [markdown]
# - polars code

# %%
start_time_for_joining = time.time()

# %%
df_left_join = df2.join(df, 
                       on="Year",
                       how="left"). \
                        sort(by=["Year"], 
                             descending=True)

df_left_join.head()

# %%
df_inner_join = df.join(df2, 
                       on="Year",
                       how="inner").sort(by=["Year"], descending=True)

df_inner_join.head()

# %%
execution_time_for_joining = time.time() - start_time_for_joining
print(f"Time to join data: {execution_time_for_joining}")

# %% [markdown]
# - pandas code: `wip...`

# %% [markdown]
# - findings
#     - despite having traditional joins, it has some extra join techniques like `semi`, `anti` like `PySpark`

# %% [markdown]
# ## 2.f.ii: concatinating

# %%
df3 = pl.DataFrame(
    {
        "Year": [2020, 2021, 2022, 2023],
        "Age": [0,0,0,0],
        "Ethnic": [1, 2, 3, 4],
        "Sex": [1, 2, 1, 2], 
        "Area": [1, 2, 3, 4],
        "count": [1000, 2000, 3000, 4000]
    }
)

# %% [markdown]
# - polars code

# %%
start_time_for_concatinating = time.time()

# %%
df_concat = pl.concat([df, df3], 
                      how="vertical_relaxed")  # vertical_relaxed: best for datatype missmatched, Int32 -> Int64

df_concat.sort(by='Year', 
                      descending=True).head(5)

# %%
execution_time_for_concatinating = time.time() - start_time_for_concatinating
print(f"time to concat data: {execution_time_for_concatinating}")

# %% [markdown]
# - pandas code: `wip...`

# %% [markdown]
# 


