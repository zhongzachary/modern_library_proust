# %%
from cmath import e
from operator import sub

import polars as pl
from polars.lazyframe.frame import T

# %%
# read the raw guide as lines_df
# each entry is separated by 2 newlines ("\n\n")
with open("../raw_text/a_guide_to_proust.md") as f:
    lines = f.read().split("\n\n")

# create df by lines
lines_df_1 = (
    pl.DataFrame({"line": lines})
    .with_columns(line=pl.col("line").str.strip_chars())
    .filter(pl.col("line") != "")
)

# assign section names to each line
lines_df_2 = (
    lines_df_1.with_columns(
        section=pl.when(pl.col("line").str.starts_with("#"))
        .then("line")
        .otherwise(None)
        .fill_null(strategy="forward")
    )
    .filter(
        pl.col("line") != pl.col("section"), pl.col("section") != pl.lit("# Foreword")
    )
    .with_columns(section=pl.col("section").str.replace(r"^# ", ""))
)

# split grouped lines into separate rows
# grouped lines are lines that start with "_" and contain multiple lines separated by "\n"
lines_df = (
    lines_df_2.with_row_index("group_index")
    .with_columns(
        is_grouped=pl.col("line").str.starts_with("_"),
        lines=pl.col("line").str.split("\n"),
    )
    .with_columns(
        group_lines=pl.when("is_grouped")
        .then(
            pl.struct(
                group=pl.col("lines").list.get(0),
                line=pl.col("lines").list.slice(1),
            )
        )
        .otherwise(
            pl.struct(
                group=None,
                line=pl.concat_list(pl.col("lines").list.join(" ")),
            )
        )
    )
    .drop("line")
    .unnest("group_lines")
    .explode("line", empty_as_null=True)
    .drop("lines")
)

# %%
# identify the entry name and the references in each line
abbreviations = ["A.J", "A. J", "Mlle.", "M.", "Mme.", "c.", "cf.", "q.v.", ".)", "..."]


def split_avoiding_abbreviations(expr, abbreviations, placeholder="_PLACEHOLDER_"):
    # replace dots in abbreviations with placeholders
    mapping = {abbr: abbr.replace(".", placeholder) for abbr in abbreviations}
    for abbr, masked_abbr in mapping.items():
        expr = expr.str.replace_all(abbr, masked_abbr, literal=True)

    # split the line by dot, preserving quoted text (e.g. "This is a speech.")
    expr = expr.str.split(r'\."?', literal=False, inclusive=True)

    # replace placeholders with back to dots
    expr = expr.list.eval(pl.element().str.replace_all(placeholder, ".", literal=True))

    return expr


# in entries_df: 1 entry -> references (list of reference)
entries_df = (
    lines_df.with_columns(
        split_avoiding_abbreviations(pl.col("line"), abbreviations).alias("entry_refs")
    )
    .with_columns(
        entry=pl.col("entry_refs").list.get(0),
        entry_index=pl.int_range(pl.len(), dtype=pl.UInt32),
        references=pl.col("entry_refs")
        .list.slice(1)
        .list.join("")
        .str.strip_chars()
        .str.split(r"\d\)?(\. |$)", inclusive=True, literal=False)
        .list.eval(pl.element().str.strip_chars())
        .list.filter(pl.element() != ""),
    )
    .drop("entry_refs", "line")
)


# %%
formatted_section_df = (
    entries_df.explode("references", empty_as_null=True)
    .select(
        "section",
        "group_index",
        "entry_index",
        group=pl.when(pl.col("group").is_not_null())
        .then(pl.lit("- ") + pl.col("group") + pl.lit("\n"))
        .otherwise(pl.lit("")),
        entry=pl.when(pl.col("group").is_not_null())
        .then(pl.lit("  "))
        .otherwise(pl.lit(""))
        + pl.lit("- ")
        + pl.col("entry")
        + pl.lit("\n"),
        reference=pl.when(pl.col("group").is_not_null())
        .then(pl.lit("  "))
        .otherwise(pl.lit(""))
        + pl.lit("  - ")
        + pl.col("references")
        + pl.lit("\n"),
    )
    .group_by(
        "section", "group_index", "entry_index", "group", "entry", maintain_order=True
    )
    .agg(
        references=pl.col("reference"),
    )
    .with_columns(
        entry_detail=pl.col("entry") + pl.col("references").list.join(""),
    )
    .group_by("section", "group_index", "group", maintain_order=True)
    .agg(entry_details=pl.col("entry_detail"))
    .with_columns(
        group_detail=pl.col("group") + pl.col("entry_details").list.join(""),
    )
    .group_by("section", maintain_order=True)
    .agg(group_details=pl.col("group_detail"))
    .select(
        text=pl.lit("# ")
        + pl.col("section")
        + pl.lit("\n\n")
        + pl.col("group_details").list.join("\n")
    )
)


# %%
with open("../formatted_guide_to_proust.md", "w") as f:
    f.write("\n".join(formatted_section_df["text"]))


# %%
# a page_ref looks something like: **I** 1, **V** 320--21, 400--8, 100
volume_re = r"\*\*(I|II|III|IV|V|VI)\**\**"
page_re = r"\d+(?:--\d+)?"
page_ref_re = rf"({volume_re} )?{page_re}"

# a year_ref looks something like: (1800--93), (c. 1900--10)
year_ref_re = r"\((c\. )?\d+--\d+\)"


# %%
pages_df = (
    entries_df.with_columns(
        references=pl.concat_list(pl.col("entry"), pl.col("references"))
    )
    .explode("references", empty_as_null=True)
    .rename({"references": "reference"})
    .with_row_index("reference_index")
    .with_columns(
        fragments=pl.col("reference")
        .str.split(r"\(|\)", literal=False, inclusive=True)
        .list.eval(
            pl.when(pl.element().str.ends_with("("))
            .then(pl.element().str.strip_suffix("("))
            .otherwise(
                pl.when(pl.element().str.ends_with(")"))
                .then(pl.lit("(") + pl.element())
                .otherwise(pl.element())
            )
        )
    )
    .explode("fragments", empty_as_null=True)
    .rename({"fragments": "fragment"})
    .with_columns(
        fragment_index=pl.int_range(pl.len(), dtype=pl.UInt32),
        is_parenthesized=pl.col("fragment").str.ends_with(")"),
        page_texts=pl.col("fragment")
        .str.replace_all(year_ref_re, "")
        .str.extract_all(page_ref_re),
    )
    .explode("page_texts", empty_as_null=True)
    .rename({"page_texts": "page_text"})
    .with_columns(
        page_struct=pl.col("page_text").str.extract_groups(
            rf"(?:(?<volume>{volume_re}) )?(?<page>{page_re})"
        )
    )
    .select(
        pl.all().exclude("page_struct"),
        pl.col("page_struct").struct.field("volume"),
        page_range=pl.col("page_struct").struct.field("page").str.split("--"),
    )
    .select(
        pl.all().exclude("page_range"),
        page_start=pl.col("page_range").list.get(0),
        page_end=pl.coalesce(
            pl.col("page_range").list.get(1, null_on_oob=True), pl.lit("")
        ),
    )
    .with_columns(
        page_end_prefix_len=pl.max_horizontal(
            pl.lit(0),
            pl.col("page_start").str.len_chars().cast(int)
            - pl.col("page_end").str.len_chars().cast(int),
        )
    )
    .with_columns(
        page_end_prefix=pl.when(pl.col("page_end_prefix_len") == 0)
        .then(pl.lit(""))
        .otherwise(pl.col("page_start").str.head("page_end_prefix_len"))
    )
    .select(
        pl.all().exclude("page_end_prefix", "page_start", "page_end"),
        page_start=pl.col("page_start").str.to_integer(strict=False),
        page_end=(pl.col("page_end_prefix") + pl.col("page_end")).str.to_integer(
            strict=False
        ),
    )
)

# %%
fill_volume_pages_df = (
    pages_df.with_columns(
        volume_inference_group=pl.when("is_parenthesized")
        .then(pl.format("{entry_index}_{fragment_index}"))
        .otherwise("entry_index")
    )
    .with_columns(
        pl.col("volume")
        .fill_null(strategy="forward")
        .over("volume_inference_group")
        .fill_null(strategy="forward")
        .over("entry_index")
    )
    .group_by(
        "section",
        "group",
        "group_index",
        "entry",
        "entry_index",
        "reference",
        "reference_index",
        maintain_order=True,
    )
    .agg(
        pages=pl.struct(
            "page_text", "is_parenthesized", "volume", "page_start", "page_end"
        )
    )
    .with_columns(
        pages=pl.col("pages").list.filter(
            pl.element().struct.field("page_text").is_not_null()
        )
    )
)

fill_volume_pages_df.write_parquet("../a_guide_to_proust.parquet")
