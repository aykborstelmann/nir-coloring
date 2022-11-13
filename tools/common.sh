ROOT_DIR=${BASH_SOURCE%/*}/..

if [ -f "$ROOT_DIR/.env" ]
then
  export $(cat "$ROOT_DIR/.env" | xargs)
fi

CYCLE_GAN_DIR=$ROOT_DIR/cycle-gan
REMOTE_REPO_PATH="dev/nir-coloring"

NIR_COLORING_ENV_PREFIX=/tmp/nir-coloring
CYCLE_GAN_ENV_PREFIX=/tmp/cycle-gan


SESSION_TYPE=local
if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_TTY" ]; then
  SESSION_TYPE=remote/ssh
else
  case $(ps -o comm= -p $PPID) in
    sshd|*/sshd) SESSION_TYPE=remote/ssh;;
  esac
fi

open-files() {
  xdg-open "$ROOT_DIR/cycle-gan/results/$2/test_*/index.html"
  xdg-open "$ROOT_DIR/cycle-gan/results-large/$2/test_*/index.html"
}