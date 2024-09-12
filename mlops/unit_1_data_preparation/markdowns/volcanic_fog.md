
```curl
curl --location 'https://localhost:6789/api/runs' \ 
--header 'Authorization: Bearer 0b8c8905f1744de091f177dba1ff52e0' \
--header 'Content-Type: application/json' \ 
--header 'Cookie: lng=en' \ 
--data '{
    "run": {
        "pipeline_uuid": "predict", 
        "block_uuid": "inference",
        "variables": {
            "inputs": [
                {
                    "DOLocationID": "239",
                    "PULocationID: "236",
                    "trip_distance": 1.98
                }
            ]
        }
    }
}'
```