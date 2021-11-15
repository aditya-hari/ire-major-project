import streamlit as st

from process_model import load_enron_models, load_wapo_model, load_spooky_model, predict_enron_models, \
    unpickle_enron_dictionary, unpickle_wapo_dictionary, unpickle_spooky_dictionary, predict_wapo_model, \
    predict_spooky_model

@st.cache(allow_output_mutation=True)
def load_model():
    enron_model = load_enron_models()
    wapo_model = load_wapo_model()
    spooky_model = load_spooky_model()
    enron_label_dict = unpickle_enron_dictionary()
    wapo_label_dict = unpickle_wapo_dictionary()
    spooky_label_dict = unpickle_spooky_dictionary()
    return enron_model, wapo_model, spooky_model, enron_label_dict, wapo_label_dict, spooky_label_dict




def predictModelMethod(input):
    enron_model, wapo_model, spooky_model, enron_label_dict, wapo_label_dict, spooky_label_dict = load_model()
    print('option',  option)
    if option == 'enron':
        prediction = predict_enron_models(enron_model, option)
        index = prediction[0]['label']
        num = int(index.replace('LABEL_', ''))
        print('author', enron_label_dict[num])
        st.write('author : ' + enron_label_dict[num])
        return
    elif option == 'wapo':
        prediction = predict_wapo_model(wapo_model, option)
        index = prediction[0]['label']
        num = int(index.replace('LABEL_', ''))
        print('author', wapo_label_dict[num])
        st.write( 'author : ' + wapo_label_dict[num])
        return
    else :
        prediction = predict_spooky_model(spooky_model, input)
        index = prediction[0]['label']
        num = int(index.replace('LABEL_', ''))
        print('author', spooky_label_dict[num])
        st.write('author : ' + spooky_label_dict[num])
        return


if __name__ == '__main__':
    st.title('Author Profiling')
    input  = st.text_area('Input', value="", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None)
    option = st.selectbox('Choose Model', ('enron', 'wapo', 'spooky'))
    button = st.button('Submit', key=None, help=None, on_click=predictModelMethod(input), args=None, kwargs=None)
    st.text('output')
