from locust import HttpUser, task, between


class BTCForecastUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(3)
    def predict_btc(self):
        self.client.post("/predict", json={"forecast_days": 7})

    @task(1)
    def check_health(self):
        self.client.get("/health")