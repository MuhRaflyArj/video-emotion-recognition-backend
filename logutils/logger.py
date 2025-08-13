import csv
import os
import datetime
from flask import request

LOGS_DIR = os.path.join(os.getcwd(), 'logutils')
LOG_FILE = os.path.join(os.getcwd(), 'logutils', 'api_logs.csv')

LOG_FIELDS = [
    'timestamp', 'request_method', 'endpoint', 'status_code', 
    'latency_ms', 'client_id', 'success', 'prediction', 
    'confidence', 'error_message'
]

def init_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()
            
def log_request(status_code, latency_ms, success, prediction=None, confidence=None, error_message=None):
    init_log_file()
    
    client_id = request.headers.get('X-Client-ID')
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'request_method': request.method,
        'endpoint': request.path,
        'status_code': status_code,
        'latency_ms': latency_ms,
        'client_id': client_id,
        'success': success,
        'prediction': prediction if prediction else '',
        'confidence': confidence if confidence else '',
        'error_message': error_message if error_message else ''
    }
    
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writerow(log_entry)
        
    return log_entry

def extract_log_filters(request):
    filters = {}

    if request.args.get('start_date'):
        filters['start_date'] = request.args.get('start_date')

    if request.args.get('end_date'):
        filters['end_date'] = request.args.get('end_date')

    if request.args.get('status_code'):
        filters['status_code'] = request.args.get('status_code')

    if request.args.get('endpoint'):
        filters['endpoint'] = request.args.get('endpoint')

    if request.args.get('success'):
        filters['success'] = request.args.get('success')

    if request.args.get('client_id'):
        filters['client_id'] = request.args.get('client_id')

    return filters

def get_logs(filters={}):
    init_log_file()
    
    logs = []
    with open(LOG_FILE, mode='r', newline='') as f:
        reader = csv.DictReader(f, fieldnames=LOG_FIELDS)
        for row in reader:
            include = True
            
            if 'start_date' in filters and 'end_date' in filters:
                log_date = datetime.datetime.fromisoformat(row['timestamp'])
                start_date = datetime.datetime.fromisoformat(filters['start_date'])
                end_date = datetime.datetime.fromisoformat(filters['end_date'])
                
                if not (start_date <= log_date <= end_date):
                    include = False
                    
            if 'status_code' in filters and row['status_code'] != str(filters['status_code']):
                include = False
            
            if 'client_id' in filters and row['client_id'] != filters['client_id']:
                include = False
                
            if 'success' in filters and row['success'].lower() != filters['success'].lower():
                include = False
                
            if 'client_id' in filters and row['client_id'] != filters['client_id']:
                include = False
            
            if include:
                logs.append(row)
                
    return logs