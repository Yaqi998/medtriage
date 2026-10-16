# Medical Image Triage - static web app served by nginx.
# All inference runs in the visitor's browser (ONNX Runtime Web + WebAssembly),
# so the container only needs to serve the files in app/. No Python, no GPU.
FROM nginx:1.27-alpine

# Render routes traffic to $PORT (10000 by default). The official nginx image
# runs envsubst on /etc/nginx/templates/*.template at start-up, so ${PORT}
# in the template is filled in from the environment.
ENV PORT=10000

RUN rm -f /etc/nginx/conf.d/default.conf
COPY deploy/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY app/ /usr/share/nginx/html/

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD wget -q -O /dev/null "http://127.0.0.1:${PORT}/" || exit 1
