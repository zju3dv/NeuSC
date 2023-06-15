import datetime
import json
import cv2

def render_space(item, render_queue):
    for view_id in range(*item['view_range']):
        render_queue.append((datetime.date(*item['time']), view_id))
        
def render_time(item, render_queue):
    time_number = item['time_number']
    start_time = datetime.date(*item['time_range'][0])
    end_time = datetime.date(*item['time_range'][1])
    for i in range(time_number):
        render_time = start_time + (end_time - start_time) * i / (time_number - 1)
        render_queue.append((render_time, item['view']))

def render_space_time(item, render_queue):
    start_time = datetime.date(*item['time_range'][0])
    end_time = datetime.date(*item['time_range'][1])
    view_range = [item['view_range'][0], item['view_range'][1] + 1]
    time_number = view_range[1] - view_range[0]
    render_times = []
    for i in range(time_number):
        render_time = start_time + (end_time - start_time) * i / (time_number - 1)
        render_times.append(render_time)
    for view_id, render_time in zip(range(*view_range), render_times):
        render_queue.append((render_time, view_id))

render_func = {
    'space': render_space,
    'time': render_time,
    'space-time': render_space_time
}


def gen_render_queue(json_path):
    json_info = json.load(open(json_path))
    render_queue = []
    for k in json_info:
        render_func[k['type']](k, render_queue)
    return render_queue

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return img

if __name__ == '__main__':
    render_queue = gen_render_queue('configs/opts/5pointz.json')
    print(render_queue)