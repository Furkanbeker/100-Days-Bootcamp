import requests

get_url = "https://jsonplaceholder.typicode.com/todos/15"

get_response = requests.get(get_url)
print(f"initial: {get_response.json()}")

#put
to_do_item_15 = {
    "userId":2 ,
    "title": "put title",
    "completed": False}
put_response = requests.put(get_url, json=to_do_item_15)
print(f"with put: {put_response.json()}")

#patch
to_do_item_patch_15 = {"title": "Patch Test"}
patch_response = requests.patch(get_url , json=to_do_item_patch_15)
print(f"with patch: {patch_response.json()}")

#delete
delete_response = requests.delete(get_url)
print(f"with delete: {delete_response.json()}")
print(delete_response.status_code)