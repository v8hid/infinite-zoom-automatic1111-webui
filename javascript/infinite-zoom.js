// Function to download data to a file
function exportPrompts(cppre, p, cpsuf, np, afn, s, w, h, smpl, gs, stps, lfn, opamt, mbr, ovm, opstr, zm, fps, zs, sf, lf, bm, bc, bg, bi, filename = "infinite-zoom-prompts.json") {

    let J = { prompts: p, negPrompt: np, prePrompt: cppre, postPrompt: cpsuf, audioFileName: afn, seed: s , width: w, height: h, sampler: smpl, guidanceScale: gs, steps: stps, lutFileName: lfn, outpaintAmount: opamt, maskBlur: mbr, overmask: ovm, outpaintStrategy: opstr, zoomMode: zm, fps: fps, zoomSpeed: zs, startFrames: sf, lastFrames: lf, blendMode: bm, blendColor: bc, blendGradient: bg, blendInvert: bi}

    var file = new Blob([JSON.stringify(J,null,2)], { type: "text/csv" });
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








