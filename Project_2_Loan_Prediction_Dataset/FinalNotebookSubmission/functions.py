import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB


def plot_loan_amounts_vs_marital_status(lp_df): 
    sub_lp_df = lp_df[['Married', 'LoanAmount']]

    # Original Scatter Plot
    # plt.figure(figsize=(14,10))
    # plt.scatter(sub_lp_df['LoanAmount'], sub_lp_df['Married'], c='green', marker='p')
    # plt.title('Marital Status vs. Loan Amount Applied For', fontsize=25)
    # plt.xlabel('Loan Amount Applied For', fontsize=20)
    # plt.ylabel('Marital Status', fontsize=20)
    # plt.show()

    # Creating a Histogram instead
    married_group_loan_amount = sub_lp_df.loc[sub_lp_df['Married'] == 'Yes'].groupby(by='LoanAmount').count()
    married_loan_amounts = married_group_loan_amount.reset_index()['LoanAmount']

    not_married_group_loan_amount = sub_lp_df.loc[sub_lp_df['Married'] == 'No'].groupby(by='LoanAmount').count()
    not_married_loan_amounts = not_married_group_loan_amount.reset_index()['LoanAmount']

    not_specified_group_loan_amount = sub_lp_df.loc[sub_lp_df['Married'] == 'Not Specified'].groupby(by='LoanAmount').count()
    not_specified_loan_amounts = not_specified_group_loan_amount.reset_index()['LoanAmount']

    plt.figure(figsize=(14,10))
    plt.hist(married_loan_amounts, color='darkorange')
    plt.hist(not_married_loan_amounts, color='lightseagreen')
    plt.hist(not_specified_loan_amounts, color='plum')
    plt.title('Distribution of Loan Amounts Applied for via Marital Status', fontsize=25)
    plt.xlabel('Loan Amount Applied For', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.legend(['Married', 'Not Married', 'Not Specified'], fontsize=15)
    plt.show()

def plot_education_status_vs_credit_history(lp_df):
    sub_lp_df = lp_df[['Education', 'Credit_History']]
    graduate_total_passed_percentage = sub_lp_df.loc[(sub_lp_df['Education'] == 'Graduate') & (sub_lp_df['Credit_History'] == 1.0)].count() / sub_lp_df.loc[sub_lp_df['Education'] == 'Graduate'].count()
    non_graduate_total_passed_percentage = sub_lp_df.loc[(sub_lp_df['Education'] == 'Not Graduate') & (sub_lp_df['Credit_History'] == 1.0)].count() / sub_lp_df.loc[sub_lp_df['Education'] == 'Not Graduate'].count()
    graduate_total_failed_percentage = 1 - graduate_total_passed_percentage
    non_graduate_total_failed_percentage = 1 - non_graduate_total_passed_percentage

    plt.figure(figsize=(14,10))
    barlist1 = plt.bar(['Graduate Passed', 'Not Graduate Passed'], [graduate_total_passed_percentage[0], non_graduate_total_passed_percentage[0]])
    barlist2 = plt.bar(['Graduate Failed', 'Not Graduate Failed'], [graduate_total_failed_percentage[0], non_graduate_total_failed_percentage[0]])
    barlist1[0].set_color('r')
    barlist1[1].set_color('r')
    barlist2[0].set_color('m')
    barlist2[1].set_color('m')
    plt.title('Education Status Credit History Pass/Fail Percentage Counts', fontsize=25)
    plt.xlabel('Education Status', fontsize=20)
    plt.ylabel('Percentage Count', fontsize=20)
    plt.legend(['Passed', 'Failed'], fontsize=15)
    plt.show()

def plot_applicant_income_vs_loan_amount_applied(lp_df):
    sub_lp_df = lp_df[['ApplicantIncome', 'LoanAmount']]
    threshold = 8000
    applicant_income_loan_amount_less_than_threshold = sub_lp_df.loc[sub_lp_df['ApplicantIncome'] < threshold]
    applicant_income_loan_amount_greater_than_threshold = sub_lp_df.loc[sub_lp_df['ApplicantIncome'] >= threshold]
    plt.figure(figsize=(14,10))
    plt.scatter(applicant_income_loan_amount_less_than_threshold['ApplicantIncome'], applicant_income_loan_amount_less_than_threshold['LoanAmount'], c='mediumseagreen')
    plt.scatter(applicant_income_loan_amount_greater_than_threshold['ApplicantIncome'], applicant_income_loan_amount_greater_than_threshold['LoanAmount'], c='lightsteelblue')
    plt.title('Applicant Income vs. Loan Amount Applied For with Threshold', fontsize=25)
    plt.xlabel('Applicant Income', fontsize=20)
    plt.ylabel('Loan Amount Applied For', fontsize=20)
    plt.legend(['Income <$8000', 'Income >=$8000'])
    plt.show()

def plot_property_area_vs_loan_status(lp_df):
    sub_lp_df = lp_df[['Property_Area', 'Loan_Status']]
    property_area_approved = sub_lp_df.loc[sub_lp_df['Loan_Status'] == 'Y'].groupby(by='Property_Area').count()
    property_area_denied = sub_lp_df.loc[sub_lp_df['Loan_Status'] == 'N'].groupby(by='Property_Area').count()
    rural_approved, semiurban_approved, urban_approved = property_area_approved['Loan_Status']
    rural_denied, semiurban_denied, urban_denied = property_area_denied['Loan_Status']

    plt.figure(figsize=(14,10))
    barlist1 = plt.bar(['Rural Approved', 'Semiurban Approved', 'Urban Approved'], [rural_approved, semiurban_approved, urban_approved], width=0.4)
    barlist2 = plt.bar([0.4, 1.4, 2.4], [rural_denied, semiurban_denied, urban_denied], width=0.4)
    list(map(lambda bar: bar.set_color('teal'), barlist1))
    list(map(lambda bar: bar.set_color('plum'), barlist2))
    plt.xticks(np.arange(0, 3, 1.0) + 0.4 / 2, ['Rural', 'Semiurban', 'Urban'])
    plt.title('Count of Approved/Denied Loan Status via Property Area', fontsize=25)
    plt.xlabel('Property Area', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.legend(['Approved', 'Denied'], fontsize=15)
    plt.show()

def plot_loan_amount_vs_loan_amount_term(lp_df):
    sub_lp_df = lp_df[['LoanAmount', 'Loan_Amount_Term']]
    # Get rid of the non specified loan amount terms
    sub_lp_df = sub_lp_df.loc[sub_lp_df['Loan_Amount_Term'] != 'Not Specified']

    unique_loan_amount_terms = sub_lp_df['Loan_Amount_Term'].unique()
    unique_loan_amount_terms.sort()
    loan_amount_term_data_series = []
    for loan_amount_term in unique_loan_amount_terms:
        loan_amount_term_data_series.append(sub_lp_df.loc[sub_lp_df['Loan_Amount_Term'] == loan_amount_term]['LoanAmount'])

    plt.figure(figsize=(14,10))
    ax = sns.boxplot(data=loan_amount_term_data_series, showmeans=True)
    plt.title('Loan Amount Boxplot per Loan Amount Term', fontsize=25)
    plt.xlabel('Loan Amount Term (Months)', fontsize=20)
    plt.ylabel('Loan Amount $', fontsize=20)
    ax.set_xticklabels(unique_loan_amount_terms)
    plt.show()

def print_cross_val_score_with_fold(classifier, x_train, y_train, fold, scoring):
    print(f'Fold = {fold}')
    print(f'Cross Validation Score: {cross_val_score(classifier, x_train, y_train, cv=fold, scoring=scoring)}')

def print_confusion_matrix_and_stats(classifier, x_train, y_train, fold):
    y_train_pred_cross_value = cross_val_predict(classifier, x_train, y_train, cv=fold)
    print(f'Fold = {fold}')
    print(f'Confusion Matrix:\n {confusion_matrix(y_train, y_train_pred_cross_value)}')
    print(f'Precision Score: {precision_score(y_train, y_train_pred_cross_value)}')
    print(f'Recall Score: {recall_score(y_train, y_train_pred_cross_value)}')
    print(f'F1 Score: {f1_score(y_train, y_train_pred_cross_value)}')

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, fold):
    plt.figure(figsize=(14,10))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend(['Precision', 'Recall'], fontsize=20)
    plt.title(f'Precision and Recall vs the Decision Threshold (Fold = {fold})', fontsize=25)
    plt.xlabel('Threshold', fontsize=20)
    plt.ylabel('Percentage', fontsize=20)
    plt.show()

def compute_scores_and_plot_precision_recall_vs_threshold(classifier, x_train, y_train, fold, method):
    Y_scores = cross_val_predict(classifier, x_train, y_train, cv=fold, method=method)
    precisions, recalls, thresholds = precision_recall_curve(y_train, Y_scores)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds, fold)

def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(14,10))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Receiver Operating Characteristic Curve', fontsize=25)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate (Recall)', fontsize=20)
    plt.show()

def plot_roc_curve_and_print_roc_auc_score(classifier, x_train, y_train, fold, method):
    Y_scores = cross_val_predict(classifier, x_train, y_train, cv=fold, method=method)
    print('ROC AUC Score:')
    print(roc_auc_score(y_train, Y_scores))
    fpr, tpr, thresholds = roc_curve(y_train, Y_scores)
    plot_roc_curve(fpr, tpr)

def print_accuracy_score(y_train, y_pred):
    correct_predictions = 0
    for true, predicted in zip(y_train, y_pred):
        if true == predicted:
            correct_predictions += 1
    accuracy = correct_predictions / len(y_train)
    print(f'Accuracy Score: {accuracy}')

def get_accuracy_score(y_train, y_pred):
    correct_predictions = 0
    for true, predicted in zip(y_train, y_pred):
        if true == predicted:
            correct_predictions += 1
    accuracy = correct_predictions / len(y_train)
    return accuracy

def get_threshold_list_and_corresponding_accuracy(classifier, x_test_data, y_test_data):
    # Creating a threshold list that will be used to test various "thresholds" for logistic regression accuracy
    threshold_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,.7,.75,.8,.85,.9,.95,.99]
    accuracy_per_threshold = []
    pred_proba_df = pd.DataFrame(classifier.predict_proba(x_test_data))
    for i in threshold_list:
        y_test_pred = pred_proba_df.applymap(lambda x: 1 if x > i else 0)
        test_accuracy = get_accuracy_score(y_test_data, y_test_pred)
        accuracy_per_threshold.append(test_accuracy)
    return threshold_list, accuracy_per_threshold

def plot_accuracy_vs_threshold(threshold_list, accuracy_list):
    # Plot the accuracies over a threshold change
    plt.figure(figsize=(14,10))
    plt.plot(threshold_list, accuracy_list, c='red')
    plt.xlabel('Threshold', fontsize=20)
    plt.ylabel('Accuracy %', fontsize=20)
    plt.title('Accuracy vs. Threshold Logistic Regression', fontsize=25)
    plt.show()

def compute_testtrain_split_accuracy(X_lp_train, Y_lp_train):
    test_size_list = [.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8]
    accuracy_list = []
    for i in test_size_list:
        X_train, X_test, Y_train, Y_test = train_test_split(X_lp_train.values, Y_lp_train.values.ravel(), test_size=i)
        Y_train = Y_train.astype('int')
        log_reg = LogisticRegression()
        log_reg = log_reg.fit(X_train, Y_train)
        y_predictions = log_reg.predict(X_train)
        accuracy_list.append(get_accuracy_score(Y_train, y_predictions))
    return test_size_list, accuracy_list

def plot_testtrain_slit_vs_accuracy(test_size_list, accuracy_list):
    plt.figure(figsize=(18,12))
    plt.plot(test_size_list, accuracy_list, c='purple')
    plt.xlabel(f'Test Size (Train = 1 - Test Size)', fontsize=25)
    plt.ylabel('Accuracy %', fontsize=25)
    plt.title('Test/Train Split vs. Accuracy', fontsize=30)
    plt.show()

