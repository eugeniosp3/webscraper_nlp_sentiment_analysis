# dgphm
# libraries



def clean_data(df):
    import re
    def pull_text(record):
        """ catches these the text between html tags for the pros and cons columns using two alternative patterns ;
        the if statement within the list comprehension removes the extra quotes """
        pattern = r"<\S*>(.*?)(?=[<])|(?<=[>])\w*(?=[<])"
        r = re.findall(pattern, record)
        r = [x for x in r if x]
        return r

    def pull_text_last(record):
        """ cleans the html tags away ;
        the if statement within the list comprehension removes the quotes """
        pattern = r"(?<=[>])(.*)(?=[<])"
        r = re.findall(pattern, record)
        r = [x for x in r if x]
        return r

    def remove_html(df, columns):
        """ removes the html from the series using the pull_text function with built in regex """
        df = df.copy()
        for c in columns:
            df.iloc[:, c] = df.iloc[:, c].apply(lambda x: pull_text(x))
        return df

    def remove_html_last(df, columns):
        """ uses the pull_text_last function to remove the html tags from the columns  """
        for c in columns:
            df.iloc[:, c] = df.iloc[:, c].apply(lambda x: pull_text_last(x))
        return df

    def replace_dashes(df, columns):
        """ removes the ---- used during scraping  """
        for c in columns:
            df.iloc[:, c] = df.iloc[:, c].apply(lambda x: x.replace('----', ''))
        return df

    def rate_review(x):
        if x <= 7:
            return 0
        else:
            return 1
        
    def drop_these(df, columns):
        for io in columns:
            df.iloc[:, io] = df.iloc[:, io].apply(lambda x: str(x).replace('----', '').replace('[', '').replace(']', '').replace('\"', ''))
        return df
    
    def list2Str(lst):
        if type(lst) is list: # apply conversion to list columns
            return";".join(lst)
        else:
            return lst
        
    # function applications  
    columns_with_html = [9, 10]  # columns - 9 pros & 10 cons
    df = remove_html(df, columns_with_html)

    next_html = [1, 3, 4, 5]  # columns - 1 review_date, 3 user_role_title, 4 company_industry & 5 company_size
    df = remove_html_last(df, next_html)

    dashed_columns = [12, 14, 15]  # columns - 12 competitors_considered & 15 others_used
    df = replace_dashes(df, dashed_columns)

    df.iloc[:, 16] = df.iloc[:, 16].str.replace("Overall Satisfaction with ", '')
    df['review_feeling'] = df.iloc[:, 6].apply(lambda x: rate_review(x))
    
    df = drop_these(df, [2, 8,9,10,11,12])
    
    df['merged_columns'] = df[['review_title', 'use_case_deployment_scope','pros', 'cons','roi', 'support_rating_usability_recommendation']].values.tolist()
    df[['review_title', 'use_case_deployment_scope','pros', 'cons'
        ,'roi', 'support_rating_usability_recommendation']] = df[['review_title', 'use_case_deployment_scope','pros'
                                                                  , 'cons','roi', 'support_rating_usability_recommendation']].astype(str)

    df = df.apply(lambda x: [list2Str(i) for i in x])
    

    return df