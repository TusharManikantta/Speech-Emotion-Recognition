// json2html.js
export default function json2html(data) {
  // Start building the HTML table with the data-user attribute
  let html = `<table data-user="tusharmanikantta@gmail.com">`;
  html += `<thead><tr><th>Name</th><th>Age</th><th>Gender</th></tr></thead><tbody>`;

  // Loop through each item in the data array and add rows to the table
  data.forEach((item) => {
    html += `<tr><td>${item.Name || ""}</td><td>${item.Age || ""}</td><td>${
      item.Gender || ""
    }</td></tr>`;
  });

  // Close the table tags
  html += `</tbody></table>`;
  return html;
}
