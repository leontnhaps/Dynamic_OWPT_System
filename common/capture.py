"""M2 acquisition validation and durable image/metadata storage; no hardware imports."""
import csv
import hashlib
import json
import re
from pathlib import Path
from common.servo import number


def camera_config(data, still=False):
    result = {}
    for key, default, low, high in [('width',1920,16,2592), ('height',1080,16,1944),
                                    ('quality',95,1,100), ('fps',10,1,60)]:
        value = number(data.get(key,default),key)
        if not value.is_integer() or not low <= value <= high:
            raise ValueError(f'{key}: integer {low}..{high} required')
        result[key] = int(value)
    exposure, gain = data.get('shutter_speed'), data.get('analogue_gain')
    if (exposure is None) != (gain is None):
        raise ValueError('Specify both exposure (us) and gain, or neither')
    if exposure is not None:
        exposure = number(exposure,'shutter_speed')
        gain = number(gain,'analogue_gain')
        if not exposure.is_integer() or not 1 <= exposure <= 1000000 or not 1 <= gain <= 16:
            raise ValueError('Exposure: 1..1000000 us; gain: 1..16')
    result.update(shutter_speed=int(exposure) if exposure is not None else None,
                  analogue_gain=gain, awb=bool(data.get('awb',True)))
    return result


def capture_request(data):
    token = str(data.get('request_id',''))
    if not re.fullmatch(r'[a-f0-9]{32}',token):
        raise ValueError('Capture requires a UUID hex request_id')
    source = data.get('measurement',{})
    measurement = {}
    for key in ('distance_m','target_x_m','target_y_m'):
        measurement[key] = number(source.get(key),key)
    if measurement['distance_m'] <= 0:
        raise ValueError('Distance must be positive')
    for key in ('session_label','sample_label','note','screen_orientation'):
        measurement[key] = str(source.get(key,''))[:300]
    measurement['kind'] = source.get('kind','target')
    if measurement['kind'] not in ('laser_reference','background','target'):
        raise ValueError('Invalid capture kind')
    measurement['distance_definition'] = 'forward_z_from_reference_origin_m'
    measurement['coordinate_definition'] = 'x right, y up, z forward viewed from Tx reference pose; manually measured'
    return dict(request_id=token,requested=camera_config(data,still=True),measurement=measurement)


def save_capture(directory, jpeg, meta):
    """A unique folder with exclusive files; complete.json is the commit marker.

    A failed write leaves an incomplete folder, never overwrites an earlier sample.
    The index is a convenience; each completed sample remains self-contained.
    """
    token = str(meta.get('request_id',''))
    if not re.fullmatch(r'[a-f0-9]{32}',token):
        raise ValueError('Invalid capture ID')
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    sample = directory / token
    sample.mkdir()  # Deliberately reject duplicates.
    path = sample / 'image.jpg'
    with path.open('xb') as f:
        f.write(jpeg)
    meta = dict(meta,sha256=hashlib.sha256(jpeg).hexdigest(),image_file='image.jpg')
    with (sample/'metadata.json').open('x',encoding='utf-8') as f:
        json.dump(meta,f,ensure_ascii=False,indent=2,default=str,allow_nan=False)
    (sample/'complete.json').write_text(json.dumps({'request_id':token,'sha256':meta['sha256']}),encoding='utf-8')
    return path, meta


def append_index(directory, path, meta):
    commanded = meta.get('servo_at_capture_start',{}).get('commanded') or {}
    measurement = meta['measurement']
    row = dict(request_id=meta['request_id'],image=str(path),simulated=meta['simulated'],
               **{k:measurement[k] for k in ('session_label','sample_label','kind','distance_m','target_x_m','target_y_m')},
               pan_command_deg=commanded.get('pan'),tilt_command_deg=commanded.get('tilt'),
               ir_gpio_level=meta.get('ir_gpio_level'),laser_gpio_level=meta.get('laser_gpio_level'),
               gpio_changed_during_capture=meta.get('gpio_changed_during_capture'),
               capture_done_unix_ns=meta['capture_done_unix_ns'])
    index = Path(directory)/'index.csv'
    exists = index.exists() and index.stat().st_size > 0
    with index.open('a',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)


def export_catalog(directory):
    """Rebuild a portable CSV from committed samples and current annotations."""
    directory=Path(directory)
    fields=['request_id','image','session_label','sample_label','kind','distance_m','target_x_m','target_y_m',
            'pan_command_deg','tilt_command_deg','width','height','quality','requested_exposure_us','requested_gain',
            'ir_gpio_level','laser_gpio_level','gpio_changed_during_capture','u_L','v_L','u_target','v_target','simulated']
    rows=[]
    for marker in sorted(directory.glob('*/complete.json')):
        meta=json.loads(marker.with_name('metadata.json').read_text(encoding='utf-8'))
        annotations=marker.with_name('annotations.json')
        points=json.loads(annotations.read_text(encoding='utf-8')).get('points',{}) if annotations.exists() else {}
        measurement=meta['measurement'];cfg=meta['requested']
        commanded=meta.get('servo_at_capture_start',{}).get('commanded') or {}
        size=meta.get('actual_image_size',[cfg['width'],cfg['height']])
        row={k:measurement.get(k) for k in ('session_label','sample_label','kind','distance_m','target_x_m','target_y_m')}
        row.update(request_id=meta['request_id'],image=f'{marker.parent.name}/image.jpg',
                   pan_command_deg=commanded.get('pan'),tilt_command_deg=commanded.get('tilt'),
                   width=size[0],height=size[1],quality=cfg['quality'],requested_exposure_us=cfg['shutter_speed'],
                   requested_gain=cfg['analogue_gain'],simulated=meta['simulated'],
                   ir_gpio_level=meta.get('ir_gpio_level'),laser_gpio_level=meta.get('laser_gpio_level'),
                   gpio_changed_during_capture=meta.get('gpio_changed_during_capture'),
                   u_L=points.get('laser',{}).get('u'),v_L=points.get('laser',{}).get('v'),
                   u_target=points.get('target',{}).get('u'),v_target=points.get('target',{}).get('v'))
        rows.append(row)
    path=directory/'calibration.csv'
    temp=path.with_suffix('.tmp')
    with temp.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(path)
    return path,len(rows)
