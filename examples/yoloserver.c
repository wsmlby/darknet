#include "darknet.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

void print_detections(image im, detection *dets, int num, float thresh, char **names, int classes)
{
    int i,j;

    for(i = 0; i < num; ++i){
        char labelstr[4096] = {0};
        int class = -1;
        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j] > thresh){
                if (class < 0) {
                    strcat(labelstr, names[j]);
                    class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%%", names[j], dets[i].prob[j]*100);
            }
        }
        if(class >= 0){
            box b = dets[i].bbox;
            // printf(" %f %f %f %f::", b.x, b.y, b.w, b.h);

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;
            
            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            printf(" [%d+%d, %d+%d], %d %d\n", left, right - left, top, bot - top, im.w, im.h);
        }
    }
}

typedef struct {
    network *net;
    char **names;
    layer l;
} model;

model* create_model(char *cfgfile, char *weightfile) {
    model* m = malloc(sizeof(model));
    list *options = read_data_cfg("cfg/coco.data");
    char *name_list = option_find_str(options, "names", "data/names.list");
    m->names = get_labels(name_list);
    m->net = load_network(cfgfile, weightfile, 0);
    m-> l = m->net->layers[m->net->n-1];
    set_batch_network(m->net, 1);
    return m;
}

void run_model(model* m, char *filename, float thresh) {
    clock_t time;
    char buff[256];
    char *input = buff;
    float nms=.4;
    strncpy(input, filename, 256);
    image im = load_image_color(input,0,0);
    image sized = letterbox_image(im, m->net->w, m->net->h);


    float *X = sized.data;
    time=what_time_is_it_now();
    network_predict(m->net, X);
    printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
    int nboxes = 0;
    detection *dets = get_network_boxes(m->net, im.w, im.h, thresh, 0.5, 0, 1, &nboxes);
    if (nms) do_nms_sort(dets, nboxes, m->l.classes, nms);
    print_detections(im, dets, nboxes, thresh, m->names, m->l.classes);
    free_detections(dets, nboxes);
    free_image(im);
    free_image(sized);
}

int main(int argc, char **argv)
{    
    srand(2222222);
    model *m = create_model(argv[1], argv[2]);
    run_model(m, argv[3], 0.25);
    run_model(m, argv[4], 0.25);
    return 0;
}

