def predict_latte_price(income, rent):
    return round(2.5 + income * 0.00002 + rent * 0.001, 2)
