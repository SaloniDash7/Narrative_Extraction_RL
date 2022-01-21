# Narrative_Extraction_RL
Narrative Extraction Using Human-In-The-Loop Reinforcement Learning

## Setup

```
pip install -r requirements.txt
bash setup.sh
```

Download data from [here](https://microsoftapc-my.sharepoint.com/:f:/g/personal/t-sadash_microsoft_com/Em_k0ahbPJFMuVfJa5lVI4QBCl_FslQDNq63ir7QnQbUZQ?e=fdCfIR&xsdata=MDN8MDF8fGU2ZmQ4MjczNzE0ZjQwYjlhNGRlNGIyNDhjZjVkNzAwfDcyZjk4OGJmODZmMTQxYWY5MWFiMmQ3Y2QwMTFkYjQ3fDF8MHw2Mzc3NTkyNzQ0Njc4Nzg0OTZ8R29vZHxWR1ZoYlhOVFpXTjFjbWwwZVZObGNuWnBZMlY4ZXlKV0lqb2lNQzR3TGpBd01EQWlMQ0pRSWpvaVYybHVNeklpTENKQlRpSTZJazkwYUdWeUlpd2lWMVFpT2pFeWZRPT0%3D&sdata=Z3plVmNwQ3NTUk43OC9ROE1TZ3U5elAzWXBkM2VqOHJOWmJoOGpHUlQ0MD0%3D&ovuser=72f988bf-86f1-41af-91ab-2d7cd011db47%2Ct-kabirahuja%40microsoft.com) and place it in `data/` directory.

## Test Environment with Trivial Policies
```
python -m src.test_env
```