import json

def convert_to_mm_ss_mls(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    milliseconds = int((remaining_seconds - int(remaining_seconds)) * 1000)
    formatted_time = f"{minutes:02}:{int(remaining_seconds):02}:{milliseconds:03}"
    return formatted_time

# Load the JSON data from the file
with open('./webhook_response.json', 'r') as file:
    data = json.load(file)

# Convert start and end times
for entry in data['output']['identification']:
    entry['start'] = convert_to_mm_ss_mls(entry['start'])
    entry['end'] = convert_to_mm_ss_mls(entry['end'])

# Save the updated JSON data back to the file
with open('webhook_response_converted.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Start and end times have been converted and saved to webhook_response_converted.json")
