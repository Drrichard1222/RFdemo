# 导入 Streamlit 库，用于构建Web 应用
import streamlit as st

# 导入joblib库，用于加载和保存机器学习模型
import joblib

# 导入NumPy库，用于数值计算
import numpy as np

# 导入Pandas库，用于数据处理和操作
import pandas as pd

# 导入SHAP库，用于解释机器学习模型的预测
import shap

# 导入Matplotlib库，用于数据可视化
import matplotlib. pyplot as plt

#从LIME 库中导入LimeTabuLarExpLainer,用于解释表格数据的机器学习模型
from lime.lime_tabular import LimeTabularExplainer

#加载训练好的随机森林模型（RF.pkL）
model = joblib.load('RF.pkl')

#从X_test.csv文件加载测试数据，以便用于LIME解释器
X_test = pd.read_csv('X_test.csv')

#定义特征名称，对应数据集中的列名
feature_names = [
    "X_30", # X_30
    "X_39", # X_39
    "X_46", # X_46
    "X_32", # X_32
    "X_34", # X_34
    "X_33", # X_33
    "X_9", # X_9
    "X_28" # X_28
]

# Streamlit用面
st.title("XX病风险预测器") # 设置网页标题
         
# X_30：数值输入框
X_30 = st.number_input("X_30:", min_value=0, max_value=500, value=41)

# X_39：数值输入框
X_39 = st.number_input("X_39:", min_value=0, max_value=15000, value=41)

# X_46：数值输入框
X_46 = st.number_input("X_46:", min_value=0, max_value=100, value=41)

# X_32：数值输入框
X_32 = st.number_input("X_32:", min_value=0, max_value=10, value=4.1)

# X_34：数值输入框
X_34 = st.number_input("X_34:", min_value=0, max_value=200, value=41)

# X_33：数值输入框
X_33 = st.number_input("X_33:", min_value=0, max_value=100, value=41)

# X_9：数值输入框
X_9 = st.selectbox("X_9:", options=[0, 1], format_func=lambda x:"0"if x==1 else"1")

# X_33：数值输入框
X_28 = st.number_input("X_28:", min_value=0, max_value=100, value=41)

# 处理输入数据并进行预测
feature_values = [X_30, X_39, X_46, X_32, X_32, X_34, X_33, X_9, X_28]# 将用户输入的特征值存入列表
features=np.array([feature_values])# 将特征转换为NumPy 数组，适用于模型输入

# 当用户点击“ Predict”按钮时执行以下代码
if st. button("Predict"):
    # 预测类别（0：无XX病，1：有XX病）
    predicted_class = model.predict(features)[0] # 预测类别的概率
    predicted_proba = model. predict_proba(features)[0] 
    # 显示预测结果
    st.write(f"**Predicted Class: * {predicted_class} (1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities: ** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为1（高风险）
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of XX disease. "
            f"The model predicts that your probability of having XX disease is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    # 如果预测类别为0（低风险）
    else:
        advice = (
            f"According to our model, you have a low risk of XX disease."
            f"The model predicts that your probability of not having XX disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    # 显示建议
    st.write(advice)

    # SHAP解释
    st.subheader("SHAP Force Plot Explanation") 
    #创建SHAP解释器，基于树模型（如随机森林）
    explainer_shap = shap.TreeExplainer(model) 
    #计算SHAP值，用于解释模型的预测
    shap_values = explainer_shap. shap_values(pd. DataFrame([feature_values], columns=feature_names)) 
    
    # 根据预测类别显示SHAP强制图
    # 期望值（基线值）
    # 解释类别1（患病）的SHAP值
    # 特征值数据
    # 使用Matplotlib 绘图
    if predicted_class == 1:
        shap.force_plot(explainer_shap. expected_value[1], shap_values[1], pd. DataFrame([feature_values], columns=feature_names), matplotlib=True) 
    # 期望值（基线值）
    # 解释类别0（未患病）的SHAP值
    # 特征值数据
    # 使用Matplotlib绘图
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True) 
    
    plt. savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200) 
    st. image("shap_force_plot. png", caption='SHAP Force Plot Explanation')
    
    
    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Not sick', 'Sick'], # Adjust class names to match your classification task mode='classification
)

    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features. flatten(),
        predict_fn=model.predict_proba
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False) # Disable feature value table
    st.components.v1.html(lime_html, height=800, scrolling=True)  