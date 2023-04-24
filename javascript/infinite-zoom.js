// Function to download data to a file
function exportPrompts(cp,p, np, filename = "infinite-zoom-prompts.json") {

    let J = { prompts: p, negPrompt: np, commonPrompt: cp }

    var file = new Blob([JSON.stringify(J)], { type: "text/csv" });
    if (window.navigator.msSaveOrOpenBlob) // IE10+
        window.navigator.msSaveOrOpenBlob(file, filename);
    else { // Others
        var a = document.createElement("a"),
            url = URL.createObjectURL(file);
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(function () {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 0);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const onload = () => {

        if (typeof gradioApp === "function") {
            /* move big buttons directly under the tabl of prompts as SMall ones */
            const wrap = gradioApp().querySelector("#tab_iz_interface .gradio-dataframe .controls-wrap")

            if (wrap) {
                let butts = gradioApp().querySelectorAll("#tab_iz_interface .infzoom_tab_butt")
                butts.forEach(b => {
                    wrap.appendChild(b)
                    b.classList.replace("lg", "sm") // smallest
                });
            }
            else {
                setTimeout(onload, 2000);
            }
        }
        else {
            setTimeout(onload, 2000);
        }
    };
    onload();
});








