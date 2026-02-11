# Lecture Report of the Advanced Applied Research IV

## Usage

1. `git clone git@github.com:kktsuji/advanced-applied-research-4.git` on WSL
2. `cd advanced-applied-research-4`
3. `docker build -t pandoc/japanese .` to build the Docker image
4. Execute following command to generate PDF file from Markdown file:

```bash
docker run --rm \
       --volume "$(pwd):/data" \
       --user $(id -u):$(id -g) \
       pandoc/japanese docs/report.md -o docs/report.pdf \
       --pdf-engine=lualatex -V documentclass=ltjsarticle
```
