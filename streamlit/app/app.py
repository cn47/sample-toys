import pandas as pd
import streamlit as st
from multiapp import MultiApp
from PIL import Image

st.set_page_config(
    page_title="Sample App!!",
    page_icon="🦈",
    layout="wide",
    initial_sidebar_state="expanded",
)


data_csv = "./data/newly_confirmed_cases_daily.csv"
df = pd.read_csv(data_csv)
df["Date"] = pd.to_datetime(df.Date)

### Define Process #############################################################
def main():
    app = MultiApp()
    app.add_app("画像1", さんぷる)
    app.add_app("画像2", もぐら)
    app.add_app("コロナDF-東京/北海道/大阪", corona_positive_dataframe)
    app.add_app("コロナ線グラフ-東京/北海道/大阪", corona_positive_line)
    app.add_app("コロナ散布図-東京/大阪", corona_positive_scatter)
    app.run()


### Define Function ############################################################
def さんぷる():
    img = "./data/Screenshot_20220120-005008999.jpg"
    caption = "ozex"
    image = Image.open(img)
    st.image(image, caption=caption, use_column_width=True)


def もぐら():
    img = "./data/Screenshot_20220212-084733242.jpg"
    caption = "mole"
    image = Image.open(img)
    st.image(image, caption=caption, use_column_width=True)


###
def corona_positive_dataframe():
    st.dataframe(df.iloc[:, 0:], width=None, height=700)


###
def corona_positive_line():
    with st.echo(code_location="below"):
        st.line_chart(df[["Tokyo", "Hokkaido", "Osaka"]], width=150, height=700)


###
def corona_positive_scatter():
    with st.echo(code_location="below"):
        fig = px.scatter(x=df.Date, y=df.Tokyo)
        fig.add_scatter(x=df.Date, y=df.Osaka, mode="markers")
        fig.update_layout(
            xaxis_title="日付", yaxis_title="都道府県", font_size=15, hoverlabel_font_size=20, width=1000, height=700
        )
        st.write(fig)


### Execute Process ############################################################
if __name__ == "__main__":
    main()
