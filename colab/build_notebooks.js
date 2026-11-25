// Convert the self-contained cell sources to Colab-uploadable notebooks.
const fs = require('fs');
const path = require('path');
for (const [source, target] of [
  ['real_pdf_workflow.py', 'Real_PDF_LLM_Defence_Colab.ipynb'],
  ['train_classifier_t4.py', 'Train_Transformer_Classifier_T4.ipynb']
]) {
  const parts = fs.readFileSync(path.join(__dirname, source), 'utf8').split(/^# %%/m).slice(1);
  const cells = parts.map((part, i) => {
    const markdown = part.startsWith(' [markdown]');
    let text = part.slice(part.indexOf('\n') + 1).trimEnd();
    if (markdown) text = text.split('\n').map(line => line.replace(/^# ?/, '')).join('\n');
    return {cell_type: markdown ? 'markdown' : 'code', id: 'cell-' + i,
      metadata: {}, source: text.split(/(?<=\n)/),
      ...(markdown ? {} : {execution_count: null, outputs: []})};
  });
  fs.writeFileSync(path.join(__dirname, target), JSON.stringify({cells, metadata: {
    colab: {name: target, provenance: []}, accelerator: 'GPU',
    kernelspec: {display_name: 'Python 3', language: 'python', name: 'python3'},
    language_info: {name: 'python'}}, nbformat: 4, nbformat_minor: 5}, null, 1)+'\n');
  console.log(target, cells.length, 'cells');
}
