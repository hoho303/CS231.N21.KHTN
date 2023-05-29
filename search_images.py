import os
import pandas as pd
import streamlit as st
from typing import List
from clip_model import CLIPOpenAIFaiss

from datetime import datetime
from constants import DATA_DIR, INDEX_LOOKUP_FILE
from visualization import read_image

if "DATA_SELECTION" not in st.session_state:
    st.session_state["DATA_SELECTION"] = {
        "Oxford 102 Flowers": "ox_102_flowers/image_index.csv",
        "Oxford Building": "oxbuild_images/image_index.csv",
        "Paris Building": "paris_images/image_index.csv",
    }

# if "MODEL_SELECTION" not in st.session_state:
#     st.session_state["MODEL_SELECTION"] = {
#         "OpenAICLIP-FasterImage+FAISS": CLIPOpenAIFaiss(
#             INDEX_LOOKUP_FILE, None, k_neighbors=5
#         )
#     }

class Result:
    def __init__(self, image, score) -> None:
        self.image = image
        self.score = score


def read_csv(csv_name):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
    return df


def get_results(
    df, clip_model, query, image = None, score_thresh=20.0, top_k=5
) -> List[Result]:
    results = []

    # use CLIP model to get similarity scores and pick the top_k
    if image is not None:
        df_output = clip_model.get_similarity(df["image_path"].values, image)
    else:
        df_output = clip_model.get_similarity_scores(df["image_path"].values, query)

    df_output = df_output.sort_values("score", ascending=False)
    df_output = df_output[df_output["score"] > score_thresh]

    for _, row in df_output.iterrows():
        image = read_image(row["image_path"])
        result = Result(image, row["score"])
        results.append(result)
        if len(results) >= top_k:
            break
    return results


def show_results(results: List[Result], time_elapsed, top_k=3):
    top_k = min(top_k, len(results))
    st.write(f"Found {len(results)} results. Showing top {top_k} results below: ")
    count = 0
    for result in results:
        image = result.image
        caption = f"Score {result.score:.2f}"
        st.image(
            image,
            caption=caption,
            width=None,
            use_column_width=None,
            clamp=False,
            channels="RGB",
            output_format="auto",
        )
        count += 1
        if count >= top_k:
            break


def main():
    st.title("Image Search App")
    st.write("This app finds similar images to your query.")
    data_selection = st.selectbox(
        label="Dataset",
        options=st.session_state["DATA_SELECTION"].keys(),
    )

    # model_selection = st.selectbox(
    #     label="Model",
    #     options=st.session_state["MODEL_SELECTION"].keys(),
    # )

    clip_model = CLIPOpenAIFaiss(os.path.join(DATA_DIR, st.session_state["DATA_SELECTION"][data_selection]), None, k_neighbors=5)
    df = read_csv(st.session_state["DATA_SELECTION"][data_selection])
    start_time = datetime.now()    
    query = st.text_input("Search Query", "")
    image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    results = get_results(df, clip_model, query, image)
    time_elapsed = datetime.now() - start_time

    show_results(results, time_elapsed.total_seconds())


if __name__ == "__main__":
    main()