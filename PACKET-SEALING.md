# Packet sealing and number OCR

Start Jewel Guard with `./run_tracker_slic.sh`. Finish a capture attempt for every jewel, then
return the jewels separately to the bench ROI. Click **Packet Sealing + OCR**.
The button waits for identity preparation and active capture to finish. Finalized
INCOMPLETE attempts and captured attempts still being analyzed can proceed.
Untested jewels still block the button. The text beside the button names the blocker.

After camera/stone tracking stops, pending testing analysis and report jobs drain
in the background while packing starts. Testing models stay loaded until those
jobs finish, then are released. The gold model uses its shared inference dispatcher
for both recorded recovery and packing until recovery ends. The packet window reuses the
same jewel tracks, IDs, identity gallery and loaded gold detector. The camera is
released and reconnected with the same source/settings during the handoff.
Keep the jewels in place while it loads. Testing results remain in the main window.

The sibling `../../packet_sealing` application supplies packet segmentation, bag
false-positive filtering, strip detection, strip filtering and hand segmentation.
Set `JEWEL_PACKET_DIR` to its full directory if installed elsewhere. It must contain
`standalone/` and `model onnx/` with `bagnewmodel.onnx`,
`fp_filter_resnet18_bag.onnx`, `stripmodel.onnx`, `hsv_fp_filter_strip.pt`,
`handseg model.onnx` and `pcnumber.onnx`. These files are existing assets.

Pack the identified jewels. Each pickup retains its ID; packet contact alone
cannot confirm placement. First validated strip detection starts the sealing
stage. Five clear, fresh absent-strip observations over at least one second
confirm strip removal with retained packet context. Sealing is inferred from
strip removal; adhesive strength and hidden contents are not measured.

After sealing, rotate the packet to show its number inside the bench ROI.
Three matching OCR reads with at least 85% confidence spanning at least one
second confirm the number and automatically save. Use **Finish / Save** to save
an incomplete session; an unconfirmed number is explicitly recorded as such.

OCR uses the packet application's isolated PaddleOCR environment. To verify it:

```bash
cd ../../packet_sealing
./run_packet_sealing.sh --check-ocr
```

If unavailable, follow `README-packet-app.md` and `setup_packet_ocr.sh` there.
`PACKET_OCR_PYTHON` may point to another compatible OCR interpreter. Number-model
and OCR preparation starts in the background when the strip is first accepted.

Output is saved inside the current testing session's `packet_sealing/` directory:
`report.pdf` includes packing, sealing, OCR and each original jewel model result;
`jewel_testing_report.pdf` preserves the full testing report; `evidence.json`
links the original IDs/results and testing directory to the packet number.
Transfer videos and the sealing/OCR video use the existing packet exporter.
The main window links to the new report when export completes.

**Reset (re-arm)** in the main window first saves any active packet session, then
starts fresh testing. Closing during export waits for evidence to finish. The
packet window disables independent recount/reset and camera/ROI changes in a
handed-off session, so it cannot silently assign new jewel IDs.

The final combined export waits for background testing analysis and report updates.
Incomplete predictions remain absent in JSON and are labeled INCOMPLETE in the PDF,
with their close reason. Sealing never changes a jewel test outcome. Testing cleanup
preserves an active `packet_sealing/` folder. Reset/close waits for both stages to save.
