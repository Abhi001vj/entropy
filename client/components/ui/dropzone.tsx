"use client";

import { cn, uploadBlob } from "@/lib/utils";
import { useEffect, useState } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { FileIcon, TrashIcon } from "@radix-ui/react-icons";
import { Icons } from "./icons";
import { FormLabel } from "./form";
import { toast } from "sonner";

function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return "0 Bytes";

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}

function useDragDrop() {
  const [dragOver, setDragOver] = useState<boolean>(false);
  const [fileDropError, setFileDropError] = useState<string>("");

  const onDragOver = (e: React.SyntheticEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const onDragLeave = () => setDragOver(false);

  return {
    // Drag
    dragOver,
    setDragOver,
    onDragOver,
    onDragLeave,
    // Errors
    fileDropError,
    setFileDropError,
  };
}

export default function Dropzone({ form }) {
  const uploadedFile = form.watch("custom_lora_file");
  const [isLoading, setIsLoading] = useState(false);

  const { dragOver, setDragOver, onDragOver, onDragLeave, setFileDropError } =
    useDragDrop();

  const handleFile = (file: File) => {
    if (file && file.name.endsWith(".safetensors")) {
      setIsLoading(true);
      uploadBlob(file)
        .then((blob) => {
          if (blob && blob.url) {
            form.setValue("custom_lora", blob.url);
            console.log("blob", blob);
          } else {
            console.error("Failed to get the URL from the uploaded blob.");
            toast.error("Failed to upload file.");
          }
          setIsLoading(false);
        })
        .catch((error) => {
          console.error("Upload failed", error);
          toast.error("Failed to upload file.");
          setIsLoading(false);
        });
    } else {
      setFileDropError("Only .safetensors files are allowed.");
      toast.error("Only .safetensors files are allowed.");
    }
  };

  const onDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setDragOver(false);

    const selectedFiles = e.dataTransfer.files;
    if (selectedFiles.length) {
      handleFile(selectedFiles[0]);
    }
  };

  const fileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) {
      return setFileDropError("No file selected!");
    }

    const file = files[0];
    console.log("file", file);
    form.setValue("custom_lora_file", file);
    setFileDropError("");
    handleFile(file);
  };

  const handleDelete = () => {
    form.setValue("custom_lora", "");
    form.setValue("custom_lora_file", null);
  };

  // useEffect(() => {
  //   if (uploadedFile) handleFile(uploadedFile);
  // }, [uploadedFile]);

  return (
    <>
      {/* Uploader */}
      <div className="dark:bg-muted bg-white w-full max-w-lg rounded-xl">
        {!uploadedFile ? (
          <form>
            <label
              htmlFor="file"
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
            >
              <div
                className={cn(
                  "px-4 py-4 h-full border-[1.5px] border-dashed dark:border-neutral-700 rounded-xl flex flex-col items-center hover:cursor-pointer",
                  dragOver && "border-blue-600 bg-blue-50"
                )}
              >
                <div className="h-full flex flex-col justify-start items-center gap-2">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="icon icon-tabler icon-tabler-cloud-upload h-6 w-6"
                    width="44"
                    height="44"
                    viewBox="0 0 24 24"
                    strokeWidth="1.5"
                    stroke={dragOver ? "#3b82f6" : "#525252"}
                    fill="none"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path stroke="none" d="M0 0h24v24H0z" fill="none" />
                    <path d="M7 18a4.6 4.4 0 0 1 0 -9a5 4.5 0 0 1 11 2h1a3.5 3.5 0 0 1 0 7h-1" />
                    <path d="M9 15l3 -3l3 3" />
                    <path d="M12 12l0 9" />
                  </svg>
                  <FormLabel>Drag and drop your own LoRA</FormLabel>
                  <p className="text-neutral-500 text-center text-xs">
                    Only .safetensors files.
                    <br /> Up to 50 MB.
                  </p>
                </div>
              </div>
            </label>
            <input
              type="file"
              name="file"
              id="file"
              className="hidden"
              onChange={fileSelect}
            />
          </form>
        ) : (
          <div className="w-full gap-2 flex flex-col justify-start items-center dark:border-neutral-700 max-h-52 overflow-auto">
            <div className="w-full flex flex-row justify-end items-center"></div>
            <div className="flex flex-row justify-between items-center border dark:border-neutral-700 gap-2 rounded-lg px-3 py-2 w-full group">
              <div className="flex flex-row w-full justify-start items-center gap-3">
                <div>
                  {isLoading ? (
                    <div className="flex flex-row justify-center items-center gap-2 h-10 w-10 border rounded-md">
                      <Icons.spinner className="h-4 w-4 animate-spin text-neutral-500" />
                    </div>
                  ) : (
                    uploadedFile && (
                      <FileIcon className="h-6 w-6 text-neutral-500" />
                    )
                  )}
                </div>
                <div className="flex flex-col justify-start items-start">
                  <div className="flex flex-row justify-start items-center gap-2">
                    <div className="max-w-[300px] truncate">
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <p className="truncate text-sm">
                              {uploadedFile.name}
                            </p>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>{uploadedFile.name}</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                  </div>
                  <div className="flex flex-row justify-start items-center gap-2">
                    <p className="text-xs text-neutral-500">
                      {formatBytes(uploadedFile.size)}
                    </p>
                    {!isLoading && (
                      <div className="flex flex-row justify-start items-center text-xs rounded-full px-2 py-[0.5px] gap-1">
                        <div className="h-2 w-2 bg-green-400 rounded-full" />
                        <p className="text-neutral-500">Uploaded</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              <div className="flex flex-row justify-end items-center gap-2">
                <button
                  className="text-neutral-400 hidden group-hover:flex flex-row justify-end bg-neutral-100 p-1.5 rounded-lg hover:text-red-500 transition-all hover:cursor-pointer"
                  onClick={() => handleDelete()}
                >
                  <TrashIcon className="h-5 w-5" />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
