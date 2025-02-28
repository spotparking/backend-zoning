import json

def load_test_set(label:str) -> list[dict]:
    with open(f"tests/test_sets/{label}.json", "r") as f:
        return json.load(f)
    
def get_enter_videos(test_set:list[dict]) -> list[str]:
    return [
        e['enter']
        for e in test_set
        if (
            e['leave'] is None and
            e['enter'] is not None
        )
    ]
    
def get_leave_videos(test_set:list[dict]) -> list[str]:
    return [
        e['leave']
        for e in test_set
        if (
            e['leave'] is not None and
            e['enter'] is not None
        )
    ]
    
def get_video_pairs(test_set:list[dict]) -> list[str]:
    return [
        (e['enter'], e['leave'])
        for e in test_set
        if (
            e['leave'] is not None and
            e['enter'] is not None
        )
    ]