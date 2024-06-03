const toolbarOptions = [
    ['bold', 'italic', 'underline', 'strike'],        // toggled buttons
    ['blockquote', 'code-block'],
    ['link', 'image', 'video', 'formula'],
   // [['code-block']] 
    [{ 'header': 1 }, { 'header': 2 }],               // custom button values
    [{ 'list': 'ordered'}, { 'list': 'bullet' }, { 'list': 'check' }],
    [{ 'script': 'sub'}, { 'script': 'super' }],      // superscript/subscript
    [{ 'indent': '-1'}, { 'indent': '+1' }],          // outdent/indent
    [{ 'direction': 'rtl' }],                         // text direction
  
    [{ 'size': ['small', false, 'large', 'huge'] }],  // custom dropdown
    [{ 'header': [1, 2, 3, 4, 5, 6, false] }],
  
    [{ 'color': [] }, { 'background': [] }],          // dropdown with defaults from theme
    [{ 'font': [] }],
    [{ 'align': [] }],
  
    ['clean']                                         // remove formatting button
  ];
  
  const quill = new Quill('#editor', {
    modules: {
         syntax: true,              // Include syntax module
      toolbar: toolbarOptions
    },
    theme: 'snow'
  });


  document.getElementById('downloadBtn').addEventListener('click', function() {
    // Get the plain text from the Quill editor
    var editorContent = quill.getText();
    
    // Create a blob of the plain text
    var blob = new Blob([editorContent], { type: 'text/plain' });
    
    // Create a link element
    var a = document.createElement('a');
    
    // Create a URL for the blob and set it as the href attribute
    var url = URL.createObjectURL(blob);
    a.href = url;
    
    // Set the download attribute with a filename
    a.download = 'content.txt';
    
    // Append the link to the body
    document.body.appendChild(a);
    
    // Programmatically click the link to trigger the download
    a.click();
    
    // Remove the link from the document
    document.body.removeChild(a);
});
/*   const Delta = Quill.import('delta');
quill.setContents(
  new Delta()
    .insert('const language = "JavaScript";')
    .insert('\n', { 'code-block': 'javascript' })
    .insert('console.log("I love " + language + "!");')
    .insert('\n', { 'code-block': 'javascript' })
); */